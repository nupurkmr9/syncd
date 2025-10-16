import math

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from einops import rearrange
from peft import LoraConfig, get_peft_model
from torch.distributions import Beta
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from .pipeline import calculate_shift as calculate_shift_flux
from .transformer import FluxTransformer2DModelWithMasking


def apply_flux_schedule_shift(noise_scheduler, sigmas, noise):
    # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
    shift = None
    # Resolution-dependent shift value calculation used by official Flux inference implementation
    mu = calculate_shift_flux(
        (noise.shape[-1] * noise.shape[-2]) // 4,
        noise_scheduler.config.base_image_seq_len,
        noise_scheduler.config.max_image_seq_len,
        noise_scheduler.config.base_shift,
        noise_scheduler.config.max_shift,
    )
    shift = math.exp(mu)
    if shift is not None:
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    return sigmas


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.to(device=device, dtype=dtype)[0]

class SynCDDiffusion(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        regularization_prob=0.1,
        num=2,
        add_lora=False,
        rank=4,
        trainkeys='lora',
        masked=True,
        shared_attn=True,
        uniform_schedule=False,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.regularization_prob = regularization_prob
        self.num = num
        self.lora_rank = rank
        self.add_lora = add_lora
        self.trainkeys = trainkeys.split('+')
        self.torch_dtype = torch.bfloat16
        self.masked = masked
        self.shared_attn = shared_attn
        self.uniform_schedule = uniform_schedule
        self.input_key = 'images'
        self.batchkeys = ['images', 'ref_images', 'prompts', 'masks', 'maskloss', 'drop_im', 'num']

        self.setup_model()
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
            shift=3,
        )

    def setup_model(self):
        dtype = self.torch_dtype
        
        self.transformer = FluxTransformer2DModelWithMasking.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='transformer',
            torch_dtype=dtype,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='text_encoder',
            torch_dtype=dtype,
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='tokenizer',
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='vae',
            torch_dtype=dtype,
        )
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.transformer.train()
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='text_encoder_2',
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='tokenizer_2',
        )
        self.text_encoder_2.requires_grad_(False)
        
        if self.add_lora:
            target_modules=[
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                ]
            unet_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            self.transformer = get_peft_model(self.transformer, unet_lora_config)

        for name, attn_proc in self.transformer.attn_processors.items():
            attn_proc.name = name

    @torch.no_grad()
    def encode_target_images(self, images):
        images = images.to(dtype=self.torch_dtype)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def get_input(self, batch, inference=False):
        # Cast input tensors to correct dtype
        batch['regularization'] = False
        x, batch = batch[self.input_key], batch
        x = x.to(dtype=self.torch_dtype, device=self.vae.device)
        # Cast other tensors in batch
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(dtype=self.torch_dtype, device=self.vae.device)
        regularization = np.random.uniform(0, 1) < self.regularization_prob
        if regularization:
            batch['regularization'] = True
        return x, batch

    def _get_clip_prompt_embeds(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            output_hidden_states=False,
        )
        prompt_embeds = prompt_embeds.pooler_output
        return prompt_embeds

    def _get_t5_prompt_embeds(self, prompt):
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_attention_mask = text_inputs.attention_mask
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(self.text_encoder.device), output_hidden_states=False
        )[0]
        return prompt_embeds, prompt_attention_mask


    @torch.no_grad()
    def encode_prompt(self, batch, num):
        captions = batch['prompts'][::num]

        pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=captions,
            )
        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=captions,
        )
        prompt_attention_mask = torch.ones((pooled_prompt_embeds.shape[0],512), device=pooled_prompt_embeds.device, dtype=pooled_prompt_embeds.dtype)
        text_ids = torch.zeros(len(captions), prompt_embeds.shape[1], 3).to(
            device=prompt_embeds.device, dtype=prompt_embeds.dtype
        )
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds, "txt_ids": text_ids, "prompt_attention_mask":prompt_attention_mask}

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width):
        batch_size, _, channels = latents.shape
        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(
            batch_size, channels // (2 * 2), height * 2, width * 2
        )

        return latents

    def forward(self, batch):
        # Get input
        x, batch = self.get_input(batch)
        # ... training step logic ...
        loss = self.compute_loss(x, batch)
            
        return loss

    def compute_loss(self, x, batch):
        x = x.to(dtype=self.torch_dtype)
        B = x.shape[0]
        if self.num == -1:
            num = batch['num']
        else:
            num = self.num
        if batch['regularization']:
            num = 1
        
        latents = self.encode_target_images(x)
        latents = rearrange(latents, "(b n) c h w -> b c h (n w)", n=num)
        _, _, H, W = latents.shape  # W = self.num * H
        noise = torch.randn_like(latents)
        noise[:, :, :, W//num:] = latents[:, :, :, W//num:]

        # # Create a Beta distribution instance
        if not self.uniform_schedule:
            beta_dist = Beta(4.0,2.0)
            sigmas = beta_dist.sample((B // num,)).to(self.transformer.device)
        else:
            sigmas = torch.rand((B // self.num,)).to(self.transformer.device)

        sigmas = apply_flux_schedule_shift(
            self.noise_scheduler, sigmas, noise
        )
        timesteps = sigmas * 1000.0
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        prompt_dict = self.encode_prompt(batch, num)
        prompt_dict = {key:value.to(latents.dtype) for key, value in prompt_dict.items()}

        noisy_latents = rearrange(noisy_latents, "b c h (n w) -> (b n) c h w", n=num)
        noisy_latents = self._pack_latents(noisy_latents, B, noisy_latents.shape[1], H, W // num)
        H1 = int(math.sqrt(noisy_latents.shape[1]))
        W1 = H1
        img_ids = prepare_latent_image_ids(B // num, H, W, self.transformer.device, self.torch_dtype)

        added_cond_kwargs = {'txt_ids': prompt_dict['txt_ids'],
                             'guidance': torch.tensor(1., device=self.transformer.device).expand(B // num)
                            }
        noisy_latents = rearrange(noisy_latents, "(b n) (h w) c -> b n h w c", n=num, h=H1, w=W1)
        noisy_latents = rearrange(noisy_latents, "b n h w c -> b (h n w) c", n=num)
        model_pred = self.transformer(
            hidden_states=noisy_latents.to(dtype=self.torch_dtype),
            timestep=timesteps.to(dtype=self.torch_dtype)/1000.,
            encoder_hidden_states=prompt_dict['prompt_embeds'].to(dtype=self.torch_dtype),
            pooled_projections=prompt_dict['pooled_prompt_embeds'].to(dtype=self.torch_dtype),
            **{x: v.to(dtype=self.torch_dtype) for x,v in added_cond_kwargs.items()},
            img_ids=img_ids.to(dtype=self.torch_dtype),
            joint_attention_kwargs={
                'attention_mask': batch['masks'].to(dtype=self.torch_dtype) if self.masked else None,
                'timestep': timesteps[0].to(dtype=self.torch_dtype)/1000,
                'shared_attn': self.shared_attn,
                'num': num
            }
        )[0]
        model_pred = rearrange(model_pred, "b (h n w) c -> (b n) (h w) c", n=num, h=H1, w=W1)
        model_pred = self._unpack_latents(model_pred[::num], H1, W1)

        target = (noise-latents)[:, :, :H, :W // num]
        
        loss = torch.mean(
                            ((model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )

        return loss.mean()