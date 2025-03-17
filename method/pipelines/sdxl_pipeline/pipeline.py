from typing import Any, Dict, List, Optional, Tuple, Union

import diffusers
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    retrieve_timesteps,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class RefAttnProc(torch.nn.Module):
    def __init__(self, attn_op, selfattn=True, name=None,):
        super().__init__()
        self.attn_op = attn_op
        self.selfattn = selfattn
        self.name = name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        self_attn_mask: Optional[torch.Tensor] = None,
        shared_attn: bool = False,
        num: int = 3,
        val: bool = False,
        mode: str = "w",
        ref_dict: dict = None,
    ):
        if self.selfattn and shared_attn:
            hw = hidden_states.shape[1]
            H = W = int(np.sqrt(hw))
            if mode == 'w':
                ref_dict[self.name] = hidden_states.detach()
                return self.attn_op(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, temb=temb, scale=scale)
            elif mode == 'r':
                hidden_states_cached = ref_dict.pop(self.name)
                hidden_states = rearrange(torch.cat([rearrange(hidden_states, "b ... -> b 1 ..."),
                                          rearrange(hidden_states_cached, "(b n) ... -> b n ...", n=num-1)], dim=1), "b n hw c -> b (n hw) c", n=num).contiguous()

            if self_attn_mask is not None and mode == 'r':
                if 'attention_mask' in ref_dict and val and H in ref_dict['attention_mask']:
                    attention_mask = ref_dict['attention_mask'][H]
                else:
                    attention_mask = F.interpolate(self_attn_mask, size=(H, W), mode="nearest")
                    attention_mask = rearrange(attention_mask, "(b n) c h w-> b (n h w) c", b=hidden_states.shape[0], n=num, h=H, w=W, c=1)
                    attention_mask = torch.einsum("b i d, b j d -> b i j", torch.ones_like(attention_mask[:, :hw]), attention_mask)
                    attention_mask[:, :hw, :hw] = 1

                    _MASKING_VALUE = -65504.  # torch.finfo(hidden_states.dtype).min
                    attention_mask = attention_mask.masked_fill(attention_mask == 0, _MASKING_VALUE).detach()
                    attention_mask = rearrange(attention_mask.unsqueeze(0).expand(attn.heads, -1, -1, -1), "nh b ... -> b nh ...")
                    if 'attention_mask' in ref_dict:
                        ref_dict['attention_mask'][H] = attention_mask
                    else:
                        ref_dict['attention_mask'] = {}
                        ref_dict['attention_mask'][H] = attention_mask

        if self.selfattn and shared_attn:
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, seqlen, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)[:, :hw]

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask,)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states[:, :hw]
        else:
            return self.attn_op(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, temb=temb, scale=scale)


def normalized_guidance_image(pred_uncond, pred_cond, pred_img, guidance_scale, img_scale):
    diff_img = pred_img - pred_uncond
    diff_txt = pred_cond - pred_img

    diff_norm_txt = diff_txt.norm(p=2, dim=[-1, -2, -3], keepdim=True)
    diff_norm_img = diff_img.norm(p=2, dim=[-1, -2, -3], keepdim=True)
    min_norm = torch.minimum(diff_norm_img, diff_norm_txt)
    diff_txt = diff_txt * torch.minimum(torch.ones_like(diff_txt), min_norm / diff_norm_txt)
    diff_img = diff_img * torch.minimum(torch.ones_like(diff_txt), min_norm / diff_norm_img)

    pred_guided = pred_img + img_scale * diff_img + guidance_scale * diff_txt
    return pred_guided


class SDXLCustomPipeline(diffusers.StableDiffusionXLPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        ###
        global_condition_type='ip_adapter',
        ip_adapter_scale=1.0,
        ip_adapter_name='ip-adapter-plus_sdxl_vit-h.bin',
        set_adapter=False,
    ):
        super().__init__(vae=vae,
                         text_encoder=text_encoder,
                         text_encoder_2=text_encoder_2,
                         tokenizer=tokenizer,
                         tokenizer_2=tokenizer_2,
                         unet=unet,
                         scheduler=scheduler,
                         image_encoder=image_encoder,
                         feature_extractor=feature_extractor,
                         force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
                         add_watermarker=add_watermarker)

        self.global_condition_type = global_condition_type
        if set_adapter:
            if 'vit-h' in ip_adapter_name:
                self.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=ip_adapter_name, image_encoder_folder="models/image_encoder")
            else:
                self.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=ip_adapter_name)

        self.set_ip_adapter_scale(ip_adapter_scale)
        if set_adapter:
            unet_lora_attn_procs = dict()
            for name, attn_proc in self.unet.attn_processors.items():
                if global_condition_type == 'ip_adapter':
                    default_attn_proc = attn_proc
                else:
                    default_attn_proc = AttnProcessor2_0()
                selfattn = True if name.endswith("attn1.processor") else False
                unet_lora_attn_procs[name] = RefAttnProc(default_attn_proc, selfattn=selfattn, name=name)

            self.unet.set_attn_processor(unet_lora_attn_procs)

    def prepare_ip_adapter_image_embeds(self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance, image_guidance_scale):
        image_embeds = ip_adapter_image_embeds[0]
        if do_classifier_free_guidance:
            negative_image_embeds = ip_adapter_image_embeds[1]

        single_image_embeds = torch.cat([image_embeds] * num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            if image_guidance_scale > 0:
                single_negative_image_embeds = torch.cat([negative_image_embeds] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds, single_image_embeds], dim=0)
            else:
                single_negative_image_embeds = torch.cat([negative_image_embeds] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

        single_image_embeds = single_image_embeds.to(device=device)
        return single_image_embeds

    def change_ipadapter_scale(self, scale):
        for name, attn_proc in self.unet.attn_processors.items():
            if name.endswith("attn2.processor") and self.global_condition_type == 'ip_adapter':
                attn_proc.attn_op.scale = [scale]

    def _get_image_adapter_scale(self, initial_scale, adaptive_scale, t):
        r"""
        Get the guidance scale at timestep `t`.
        """
        signal_scale = initial_scale + adaptive_scale * ((1000 - t)/1000)
        if signal_scale < 0:
            signal_scale = 0
        return signal_scale

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        ###
        latents_ref: Optional[torch.Tensor] = None,
        latents_mask: Optional[torch.Tensor] = None,
        ip_adapter_scale: float = 1.0,
        image_guidance_scale: float = 0.0,
        adaptive_image_guidance_scale: float = 0.0,
        **kwargs,
    ):

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if image_guidance_scale > 0:
            prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, negative_add_time_ids, add_time_ids], dim=0)
        elif self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            # make sure to directly pass the feature (different from diffusers)
            image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    num_images_per_prompt,
                    self.do_classifier_free_guidance,
                    image_guidance_scale
                )

        # 8. Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 10. Prepare cross attention kwargs and separate reference and target conditions
        self.cross_attention_kwargs.update({'val': True})
        self._num_timesteps = len(timesteps)
        self.change_ipadapter_scale(ip_adapter_scale)
        self.num = cross_attention_kwargs['num']
        num_repeat = (prompt_embeds.shape[0] // latents.shape[0])
        shared_attn = cross_attention_kwargs['shared_attn'] if 'shared_attn' in cross_attention_kwargs else False
        prompt_embeds_ref = rearrange(rearrange(prompt_embeds, "(b n) ... -> b n ...", n=self.num)[:, 1:], "b n ... -> (b n) ...")
        prompt_embeds = prompt_embeds[::self.num]
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = image_embeds
        added_cond_kwargs_ref = {x: rearrange(rearrange(v, "(b n) ... -> b n ...", n=self.num)[:, 1:], "b n ... -> (b n) ...") for x, v in added_cond_kwargs.items()}
        added_cond_kwargs = {x: rearrange(rearrange(v, "(b n) ... -> b n ...", n=self.num)[:, :1], "b n ... -> (b n) ...") for x, v in added_cond_kwargs.items()}

        # 11. Start sampling
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if cross_attention_kwargs is not None and shared_attn and latents_ref is not None and latents_mask is not None:
                    noise = randn_tensor(latents_ref.shape, generator=generator, device=device, dtype=latents.dtype)
                    latents = (1 - latents_mask) * self.scheduler.add_noise(latents_ref, noise, torch.tensor(t).expand(latents.shape[0])) + latents_mask * latents

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * num_repeat)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                ref_dict = {}
                # first extract reference features then use them in self-attention for the target
                self.change_ipadapter_scale(1.0)
                noise_pred_ref = self.unet(
                    rearrange(rearrange(latent_model_input, "(b n) ... -> b n ...", n=self.num)[:, 1:], "b n ... -> (b n) ..."),
                    t,
                    encoder_hidden_states=prompt_embeds_ref,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=added_cond_kwargs_ref,
                    return_dict=False,
                    cross_attention_kwargs={'shared_attn': shared_attn,  'num': self.num, 'mode': 'w', 'ref_dict': ref_dict, 'scale': 0.},
                )[0]
                self.change_ipadapter_scale(ip_adapter_scale)
                noise_pred = self.unet(
                    latent_model_input[::self.num],
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs={**self.cross_attention_kwargs, 'ref_dict': ref_dict, 'mode': 'r'},
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = rearrange(torch.cat([noise_pred.unsqueeze(1), rearrange(noise_pred_ref, "(b n) ... -> b n ...", n=self.num-1)], dim=1), "b n ... -> (b n) ...")
                # print("final cat", time.time()-st)
                # perform guidance
                if image_guidance_scale > 0:
                    img_scale = self._get_image_adapter_scale(image_guidance_scale, adaptive_image_guidance_scale, t)
                    noise_pred_uncond, noise_pred_image, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = normalized_guidance_image(noise_pred_uncond, noise_pred_text, noise_pred_image, self.guidance_scale, img_scale)

                elif self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # print("end", time.time()-st)
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        del ref_dict
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()
        return (image, )
