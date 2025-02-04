# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlPipeline,
    FluxTransformer2DModel,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from .flux_syncd import calculate_shift, retrieve_timesteps


def warpcorrespondence(correspondence, counter_cc, feats, num_heads, timestep, numref = 3):
    warp_first_second = True
    res = int(math.sqrt(feats.shape[1]))
    dtype = feats.dtype
    orig_res = 128

    kernel_tensor = torch.ones((1, 1, 3, 3)).to(feats.device).to(dtype)/(3*3)
    feats = rearrange(feats, "b (h w) c -> b c h w ", h=res, w=res, b=feats.shape[0], c=feats.shape[-1] )
    feats = torch.nn.functional.interpolate(feats, (orig_res, orig_res))#, mode='bicubic', antialias=True).float()
    mask = torch.zeros((feats.shape[0], 1, orig_res, orig_res), device=feats.device, dtype=feats.dtype)
    feats_copy = torch.zeros_like(feats)

    prev_counter = counter_cc[0].clone()
    counter_ = 1

    for j in range(numref-1):
        corresp = correspondence[prev_counter:prev_counter+counter_cc[counter_]].clone().to(feats.dtype)
        temp2 = torch.nan_to_num(torch.nn.functional.grid_sample(feats[-num_heads:], -1.*corresp[:, [3,2]].unsqueeze(0).unsqueeze(2).expand(num_heads, -1, -1, -1), align_corners=False, mode='bicubic' ))
        corresp_orig = (orig_res*(0.5 - corresp.clone()*0.5)).floor().long() 
        if corresp_orig.min() < 0 or corresp_orig.max() >= orig_res:
            return None
        
        feats_copy[j*num_heads:(j+1)*num_heads, :, corresp_orig[:,0], corresp_orig[:,1]] = temp2.squeeze(3)
        mask[j*num_heads:(j+1)*num_heads, :, corresp_orig[:,0], corresp_orig[:,1]] = 1.

        prev_counter += counter_cc[counter_]
        counter_ +=1

    if timestep < 0.9:
        # convolve the mask with a kernel to erode it a bit
        mask = (torch.clamp(torch.nn.functional.conv2d(mask, kernel_tensor, padding='same'), 0, 1) >= 0.9).to(feats.dtype)
    feats = mask * feats_copy + (1-mask)*feats
    
    if warp_first_second:
        mask_copy = torch.zeros((feats.shape[0], 1, orig_res, orig_res), device=feats.device, dtype=feats.dtype)
        corresp = correspondence[0:counter_cc[0]].clone().to(feats.dtype)
        temp2 = torch.nan_to_num(torch.nn.functional.grid_sample(feats[num_heads:2*num_heads], -1.*corresp[:, [3,2]].unsqueeze(0).unsqueeze(2).expand(num_heads, -1, -1, -1), align_corners=False, mode='bicubic' ))
        corresp_orig = (orig_res*(0.5 - corresp.clone()*0.5)).floor().long()  
        if corresp_orig.min() < 0 or corresp_orig.max() >= orig_res:
            return None

        feats_copy[:num_heads, :, corresp_orig[:,0], corresp_orig[:,1]] = temp2.squeeze(3)
        mask_copy[:num_heads, :, corresp_orig[:,0], corresp_orig[:,1]] = 1. * (1- mask[:num_heads, :, corresp_orig[:,0], corresp_orig[:,1]])
        if timestep < 0.9:
            # convolve the mask with a kernel to erode it a bit
            mask_copy = (torch.clamp(torch.nn.functional.conv2d(mask_copy, kernel_tensor, padding='same'), 0, 1) >= 0.9)*1.
        feats[:num_heads] = mask_copy[:num_heads] * feats_copy[:num_heads] + (1-mask_copy[:num_heads])*feats[:num_heads]


    feats = torch.nn.functional.interpolate( feats, (res,res), mode='nearest') # check with nearest also and mask.
    feats = rearrange(feats, "b c h w -> b (h w) c", h=res, w=res, b=feats.shape[0] )
    del temp2
    return feats.to(dtype)


class FluxControlCustomPipeline(FluxControlPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        num=3
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.num = num
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        height_concat = self.num * height

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids_concat = self._prepare_latent_image_ids(batch_size, height_concat // 2, width // 2, device, dtype)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids, latent_image_ids_concat

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        ###
        correspondence=None, 
        counter_cc=None,
        warp_thresh=0.9,
        negative_prompt=None,
        guidance_scale_real=2.5,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        
        ##########################
        nowarping = False
        ##########################

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare text embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        
        neg_prompt_embeds = None
        neg_pooled_prompt_embeds = None
        neg_text_ids = None

        if negative_prompt is not None:
            (
                neg_prompt_embeds,
                neg_pooled_prompt_embeds,
                neg_text_ids,
            ) = self.encode_prompt(
                prompt=[negative_prompt]*len(prompt),
                prompt_2=[negative_prompt]*len(prompt),
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 8

        control_image = self.prepare_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.vae.dtype,
        )

        if control_image.ndim == 4:
            control_image = self.vae.encode(control_image).latent_dist.sample(generator=generator)
            control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

            height_control_image, width_control_image = control_image.shape[2:]
            control_image = self._pack_latents(
                control_image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height_control_image,
                width_control_image,
            )

        latents, latent_image_ids, latent_image_ids_concat = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        self.joint_attention_kwargs.update({"txt_ids": text_ids, "img_ids_concat": latent_image_ids_concat})


        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                 # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if correspondence is not None and timestep[0]/1000 > warp_thresh and not nowarping:
                    # print(latents.shape, timestep[0]/1000)
                    latents1 = warpcorrespondence(correspondence, counter_cc, latents, 1,  timestep[0]/1000, numref = self.num)
                    if latents1 is None:
                        nowarping = True
                    else:
                        latents = latents1

                latent_model_input = torch.cat([latents, control_image], dim=2)
                self.joint_attention_kwargs.update({"timestep": timestep/1000})

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if negative_prompt is not None and i >=1:
                    noise_pred_neg = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=neg_pooled_prompt_embeds,
                        encoder_hidden_states=neg_prompt_embeds,
                        txt_ids=neg_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_neg + guidance_scale_real * (noise_pred - noise_pred_neg)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
