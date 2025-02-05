import kornia
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.models.embeddings import ImageProjection
from diffusers.training_utils import EMAModel
from einops import rearrange
from peft import LoraConfig

from ..util import append_dims, log_txt_as_img
from .pipeline import SDXLCustomPipeline

PIPELINECLASS = {
            "sdxl": SDXLCustomPipeline,
        }


class GenCDDiffusionBase(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path="ptx0/terminus-xl-gamma-v1",
        regularization_prob=0.1,
        num=3,
        resolution=1024,
        prediction_type='v_prediction',
        use_ema=False,
        global_condition_type=None,
        ip_adapter_scale=1.0,
        add_lora_self=False,
        add_lora_text=False,
        rank=128,
        snr_gamma=None,
        trainkeys='lora+to_k_ip+to_v_ip',
        masked=True,
        rescale_betas_zero_snr=True,
        ip_adapter_name='ip-adapter-plus_sdxl_vit-h.bin',
        shared_attn=True,
        training_scheduler='constant',
    ):
        super(GenCDDiffusionBase, self).__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.regularization_prob = regularization_prob
        self.num = num
        self.resolution = resolution
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        self.global_condition_type = global_condition_type
        self.add_lora_self = add_lora_self
        self.add_lora_text = add_lora_text
        self.trainkeys = trainkeys.split('+')
        self.torch_dtype = torch.float32
        self.masked = masked
        self.ip_adapter_scale = ip_adapter_scale
        self.ip_adapter_name = ip_adapter_name
        self.shared_attn = shared_attn
        self.input_key = 'images'
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.training_scheduler = training_scheduler

        pipeline, batchkeys = self.setup_pipeline(pretrained_model_name_or_path,
                                                  global_condition_type,
                                                  ip_adapter_scale,
                                                  ip_adapter_name,
                                                  torch_dtype=self.torch_dtype)
        self.batchkeys = batchkeys

        self.noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, rescale_betas_zero_snr=rescale_betas_zero_snr)
        print(self.noise_scheduler.config)

        # Initialize all submodules as independent modules and set requires grad
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.image_encoder = pipeline.image_encoder
        self.tokenizer = pipeline.tokenizer
        if hasattr(pipeline, 'text_encoder_2'):
            self.text_encoder_2 = pipeline.text_encoder_2
            self.tokenizer_2 = pipeline.tokenizer_2
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if hasattr(pipeline, 'text_encoder_2'):
            self.text_encoder_2.requires_grad_(False)
        if self.image_encoder is not None:
            self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Add LoRA modules to UNet
        if add_lora_text or add_lora_self:
            target_modules = []
            if add_lora_text:
                target_modules += ["attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"]
            if add_lora_self:
                target_modules += ["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"]
            unet_lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )

            self.unet.add_adapter(unet_lora_config)

        # Set requires grad to trainable parameters
        if self.global_condition_type is not None and 'encoder_hid_proj' in self.trainkeys:
            for param in self.unet.encoder_hid_proj.parameters():
                param.requires_grad_(True)
        if 'to_k_ip' in self.trainkeys and 'to_v_ip' in self.trainkeys:
            for name, attn_proc in self.unet.attn_processors.items():
                if hasattr(attn_proc.attn_op, 'to_k_ip'):
                    for param in attn_proc.attn_op.to_k_ip.parameters():
                        param.requires_grad_(True)
                    for param in attn_proc.attn_op.to_v_ip.parameters():
                        param.requires_grad_(True)

        del pipeline
        self.use_ema = use_ema
        if self.use_ema:
            ema_unet = self.unet
            self.ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

        self.register_buffer("clip_mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("clip_std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def setup_pipeline(self, pretrained_model_name_or_path, global_condition_type, ip_adapter_scale, ip_adapter_name, torch_dtype=torch.float32):
        pass

    def on_load_checkpoint(self, checkpoint):
        # for loading partial state dict.
        layers = []
        for key in self.state_dict().keys():
            add_ = True
            for keytosave in self.trainkeys:
                if keytosave in key:
                    add_ = False
            if add_:
                layers.append(key)
        for key in layers:
            checkpoint["state_dict"][key] = self.state_dict()[key]

    def get_input(self, batch, inference=False):
        # assuming unified data format, dataloader returns a dict. image tensors should be scaled to -1 ... 1 and in bchw format
        regularization = np.random.uniform(0, 1) < self.regularization_prob
        if regularization and not inference and 'images_reg' in batch:
            for key in self.batchkeys:
                batch[key] = batch[f'{key}_reg']
            batch['regularization'] = True
            return batch[self.input_key], batch
        else:
            batch['regularization'] = False
            return batch[self.input_key], batch

    @torch.no_grad()
    def encode_target_images(self, images):
        latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def preprocess(self, x):
        # for CLIP model
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224), interpolation="bicubic", align_corners=True, antialias=True,)
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.clip_mean, self.clip_std)
        return x

    @torch.no_grad()
    def encode_condition_image(self, image, drop_im):
        # for global image condition (IP adapter)
        output_hidden_state = not isinstance(self.unet.encoder_hid_proj.image_projection_layers[0], ImageProjection)
        if output_hidden_state:
            image_embeds = self.image_encoder(self.preprocess(image), output_hidden_states=output_hidden_state).hidden_states[-2]
        else:
            image_embeds = self.image_encoder(self.preprocess(image)).image_embeds
            drop_im = append_dims(drop_im, image_embeds.ndim)
            image_embeds = image_embeds * (1. - drop_im) + torch.zeros_like(image_embeds) * drop_im
        return image_embeds, output_hidden_state

    @torch.no_grad()
    def encode_prompt(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        # get input
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_dict = self.shared_step(batch, val=True)
        self.log_dict({f'val/loss{dataloader_idx}': loss_dict['loss']}, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        # We only train the additional adapter LoRA layers (remove this print statement)
        for name, param in self.unet.named_parameters():
            if param.requires_grad:
                print(name)

        params = []
        lr = self.learning_rate
        params += list(filter(lambda p: p.requires_grad, self.unet.parameters()))

        optimizer = torch.optim.AdamW(params, lr=lr)
        return {'optimizer': optimizer}


class GenCDDiffusionSDXL(GenCDDiffusionBase):
    def __init__(
        self,
        **kwargs,
    ):
        super(GenCDDiffusionSDXL, self).__init__(**kwargs)
        self.pipeline_model = "sdxl"

    def setup_pipeline(self, pretrained_model_name_or_path, global_condition_type, ip_adapter_scale, ip_adapter_name, torch_dtype=torch.float32,):
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype,)
        batchkeys = ['images', 'ref_images', 'prompts', 'mask', 'maskloss', 'original_size_as_tuple', 'crop_coords_top_left', 'target_size_as_tuple', 'drop_im']
        pipeline = SDXLCustomPipeline.from_pretrained(pretrained_model_name_or_path,
                                                      vae=vae,
                                                      torch_dtype=torch_dtype,
                                                      global_condition_type=global_condition_type,
                                                      ip_adapter_scale=ip_adapter_scale,
                                                      ip_adapter_name=ip_adapter_name,
                                                      set_adapter=True,
                                                      )
        return pipeline, batchkeys

    def change_scale(self, scale):
        for name, attn_proc in self.unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                attn_proc.attn_op.scale = [scale]

    @torch.no_grad()
    def encode_prompt(self, batch):
        prompt_embeds_list = []
        captions = batch['prompts']

        for tokenizer, text_encoder in zip([self.tokenizer, self.tokenizer_2], [self.text_encoder, self.text_encoder_2]):
            text_inputs = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",)
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False,)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    def shared_step(self, batch, val=False, timestep_ref=None):
        x, batch = self.get_input(batch)
        B = x.shape[0]
        latents = self.encode_target_images(x)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B // self.num,), device=self.device).long()
        timesteps = rearrange(timesteps.unsqueeze(1).expand(-1, self.num), "b n -> (b n)")

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        prompt_dict = self.encode_prompt(batch)
        add_time_ids = torch.cat([batch["original_size_as_tuple"], batch["crop_coords_top_left"], batch["target_size_as_tuple"]], dim=-1).to(self.device)

        unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": prompt_dict['pooled_prompt_embeds']}
        image_embeds = None
        if self.global_condition_type is not None:
            image_embeds, output_hidden_state = self.encode_condition_image(batch['ref_images'], batch['drop_im'])
            if batch['regularization'] and not output_hidden_state:
                image_embeds = torch.zeros_like(image_embeds)
            unet_added_conditions.update({"image_embeds": image_embeds})
        cross_attention_kwargs = {'shared_attn': self.shared_attn, 'num': self.num, 'self_attn_mask': batch['masks'] if self.masked else None}

        # pass the reference image arguments to get reference features without LoRA applied.
        ref_dict = {}
        _ = self.unet(
                    rearrange(rearrange(noisy_latents, "(b n) ... -> b n ...", n=self.num)[:, 1:], "b n ... -> (b n) ..."),
                    rearrange(rearrange(timesteps, "(b n) -> b n", n=self.num)[:, 1:], "b n -> (b n)"),
                    rearrange(rearrange(prompt_dict['prompt_embeds'], "(b n) ... -> b n ...", n=self.num)[:, 1:], "b n ... -> (b n) ..."),
                    added_cond_kwargs={x: rearrange(rearrange(v, "(b n) ... -> b n ...", n=self.num)[:, 1:], "b n ... -> (b n) ...") for x, v in unet_added_conditions.items()},
                    cross_attention_kwargs={'shared_attn': self.shared_attn, 'scale': 0., 'num': self.num, 'mode': 'w', 'ref_dict': ref_dict},
                    return_dict=False,
                )[0]

        # get the target image arguments
        cross_attention_kwargs.update({'ref_dict': ref_dict, 'mode': 'r' if not batch['regularization'] else 'n'})
        noisy_latents = rearrange(rearrange(noisy_latents, "(b n) ... -> b n ...", n=self.num)[:, :1], "b n ... -> (b n) ...")
        timesteps = rearrange(rearrange(timesteps, "(b n) -> b n", n=self.num)[:, :1], "b n -> (b n)")
        unet_added_conditions = {x: rearrange(rearrange(v, "(b n) ... -> b n ...", n=self.num)[:, :1], "b n ... -> (b n) ...") for x, v in unet_added_conditions.items()}
        prompt_dict['prompt_embeds'] = rearrange(rearrange(prompt_dict['prompt_embeds'], "(b n) ... -> b n ...", n=self.num)[:, :1], "b n ... -> (b n) ...")
        latents = rearrange(rearrange(latents, "(b n) ... -> b n ...", n=self.num)[:, :1], "b n ... -> (b n) ...")
        noise = rearrange(rearrange(noise, "(b n) ... -> b n ...", n=self.num)[:, :1], "b n ... -> (b n) ...")

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_dict['prompt_embeds'],
            added_cond_kwargs=unet_added_conditions,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # Get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            self.noise_scheduler.register_to_config(prediction_type=self.prediction_type)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean()
        loss_dict = {"loss": loss}
        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, N: int = 1, sample: bool = True, **kwargs,):

        pipelineclass = PIPELINECLASS[self.pipeline_model]
        pipeline = pipelineclass.from_pretrained(self.pretrained_model_name_or_path,
                                                 vae=self.vae,
                                                 unet=self.unet,
                                                 image_encoder=self.image_encoder,
                                                 torch_dtype=self.dtype,
                                                 global_condition_type=self.global_condition_type,
                                                 ip_adapter_scale=self.ip_adapter_scale,
                                                 ip_adapter_name=self.ip_adapter_name,
                                                 set_adapter=False).to(self.device)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config,  rescale_betas_zero_snr=self.rescale_betas_zero_snr)

        x, batch = self.get_input(batch)
        N = min(x.shape[0], N * self.num)
        latents = (self.encode_target_images(x)[:N])

        mask = batch['masks']
        mask = mask[:N]
        mask = rearrange(mask, "(b n) ... -> b n ...", b=mask.shape[0] // self.num, n=self.num)
        mask[:, :1] = 1.
        latents_mask = torch.zeros_like(mask)
        latents_mask[:, :1] = 1.
        latents_mask = rearrange(latents_mask, "b n ... -> (b n) ...")
        mask = rearrange(mask, "b n ... -> (b n) ...")

        log = {}
        log["masks"] = mask
        log["inputs"] = x
        log["inputs_cropped"] = batch['ref_images'][:N]

        ip_adapter_image = None
        ip_adapter_image_un = None
        if self.global_condition_type is not None:
            ip_adapter_image, output_hidden_state = self.encode_condition_image(batch['ref_images'][:N], batch['drop_im'])
            ip_adapter_image_un, output_hidden_state = self.encode_condition_image(torch.zeros_like(batch['ref_images'][:N]), batch['drop_im'])
            if not output_hidden_state:
                ip_adapter_image_un = torch.zeros_like(ip_adapter_image_un)
            images = pipeline(batch['prompts'][:N],
                              latents_ref=latents,
                              ip_adapter_image_embeds=[ip_adapter_image, ip_adapter_image_un],
                              cross_attention_kwargs={'shared_attn': self.shared_attn, 'num': self.num, 'self_attn_mask': torch.cat([torch.zeros_like(mask), mask], 0)},
                              latents_mask=latents_mask,
                              guidance_scale=7.5,
                              height=x.shape[2],
                              width=x.shape[3],
                              return_dict=False)[0]
            log["samples"] = images

        log['txt'] = log_txt_as_img(x.shape[2:], batch['prompts'], size=x.shape[2] // 20)
        del pipeline
        torch.cuda.empty_cache()
        return log
