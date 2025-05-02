import argparse
import glob
import os
import sys
from pathlib import Path

import kornia
import numpy as np
import torch
from einops import rearrange
from peft import LoraConfig
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers.models.embeddings import ImageProjection

sys.path.append('./')
from pipelines.sdxl_pipeline.pipeline import SDXLCustomPipeline

from data.data import DummyDataset

clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])


def preprocess(x, pipeline):
    # normalize to [0,1]
    x = kornia.geometry.resize(
        x,
        (224, 224),
        interpolation="bicubic",
        align_corners=True,
        antialias=True,
    )
    x = (x + 1.0) / 2.0
    # renormalize according to clip
    x = kornia.enhance.normalize(x, clip_mean.to(pipeline.device), clip_std.to(pipeline.device))
    return x


@torch.no_grad()
def encode_condition_image(image, pipeline):
    # make sure image is correctly normalized in the dataloader
    image_embeds = None
    output_hidden_state = False
    output_hidden_state = not isinstance(pipeline.unet.encoder_hid_proj.image_projection_layers[0], ImageProjection)
    if output_hidden_state:
        image_embeds = pipeline.image_encoder(preprocess(image, pipeline), output_hidden_states=output_hidden_state).hidden_states[-2]
    else:
        image_embeds = pipeline.image_encoder(preprocess(image, pipeline)).image_embeds
    return image_embeds


@torch.no_grad()
def encode_target_images(images, pipeline):
    latents = pipeline.vae.encode(images).latent_dist.sample()
    latents = latents * pipeline.vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode(latents, pipeline):
    latents = latents / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    return image


def sample(prompt, ref_images, ref_category, outdir, finetuned_path, num_images_per_prompt, inference_steps, numref, ip_adapter_scale, image_guidance_scale, adaptive_image_guidance_scale, guidance_scale, seed):
    seed_everything(seed)

    torch_dtype = torch.bfloat16
    vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch_dtype,
            )

    device = 'cuda'
    pipeline = SDXLCustomPipeline.from_pretrained("bghira/terminus-xl-gamma-v1",
                                                  vae=vae,
                                                  torch_dtype=torch_dtype,
                                                  global_condition_type='ip_adapter',
                                                  ip_adapter_scale=ip_adapter_scale,
                                                  ip_adapter_name='ip-adapter-plus_sdxl_vit-h.bin',
                                                  set_adapter=True).to(device)

    target_modules = ["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"]
    unet_lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipeline.unet.add_adapter(unet_lora_config)

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    if finetuned_path is not None:
        finetuned_path = torch.load(finetuned_path, map_location='cpu')
        unet_dict = {}
        for key, value in finetuned_path['state_dict'].items():
            if 'unet.' in key:
                unet_dict[key.replace('unet.', '')] = value
        pipeline.unet.load_state_dict(unet_dict, strict=False)
        print('loaded finetuned model')

    cat_ = Path(ref_images).stem
    os.makedirs(f'{outdir}/{cat_}', exist_ok=True)

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".JPG"])

    image_paths = [x for x in glob.glob(f'{ref_images}/*') if is_image_file(x)][:numref]
    dummydata = DummyDataset(image_paths=image_paths, prompt=prompt, num_images_per_prompt=num_images_per_prompt, cat=ref_category)
    dataloader = DataLoader(dummydata, batch_size=1, shuffle=False, num_workers=2, collate_fn=dummydata.collate_fn)

    counter = 0
    numref += 1
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _, batch in enumerate(dataloader):
            latents = encode_target_images(batch['images'].to(device), pipeline)
            batch['original_size_as_tuple'] = batch['target_size_as_tuple']
            mask = rearrange(batch['masks'].to(device), "(b n) ... -> b n ...", b=latents.shape[0] // numref, n=numref)
            masklatent = torch.zeros_like(mask)

            mask[:, :1] = 1.
            masklatent[:, :1] = 1.

            masklatent = rearrange(masklatent, "b n ... -> (b n) ...")
            mask = rearrange(mask, "b n ... -> (b n) ...")

            ip_adapter_image = encode_condition_image(batch['ref_images'].to(device), pipeline)
            ip_adapter_imageun = encode_condition_image(torch.zeros_like(batch['ref_images']).to(device), pipeline)
            if isinstance(pipeline.unet.encoder_hid_proj.image_projection_layers[0], ImageProjection):
                ip_adapter_imageun = torch.zeros_like(ip_adapter_imageun)

            self_attn_mask = torch.cat([torch.zeros_like(mask), mask, mask], 0) if image_guidance_scale > 0 else torch.cat([torch.zeros_like(mask), mask], 0)
            generated_image = pipeline(batch['prompts'],
                                       latents_ref=latents,
                                       ip_adapter_image_embeds=[ip_adapter_image, ip_adapter_imageun],
                                       cross_attention_kwargs={'shared_attn': True, 'num': numref, 'self_attn_mask': self_attn_mask},
                                       latents_mask=masklatent,
                                       guidance_scale=guidance_scale,
                                       num_inference_steps=inference_steps,
                                       return_dict=False,
                                       image_guidance_scale=image_guidance_scale,
                                       adaptive_image_guidance_scale=adaptive_image_guidance_scale,
                                       ip_adapter_scale=ip_adapter_scale,
                                       generator=torch.Generator(device="cpu").manual_seed(seed + counter))[0]
            torch.cuda.empty_cache()
            generated_image = generated_image.cpu()

            for index, img in enumerate(generated_image[::numref]):
                img = Image.fromarray(((torch.clip(img.float(), -1., 1.).permute(1, 2, 0).cpu().numpy()*0.5+0.5)*255).astype(np.uint8))
                name = f'{outdir}/{cat_}/{counter:05d}.png'
                img.save(name)
                counter += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the sampling script"
    )
    parser.add_argument("--prompt", type=str, help="prompt", default="An actionfigure wearing a hat")
    parser.add_argument("--ref_images", type=str, help="path to reference images")
    parser.add_argument("--ref_category", type=str, help="reference category e.g. toy")
    parser.add_argument("--outdir", type=str, default="./samples")
    parser.add_argument('--finetuned_path', type=str, help="path to finetuned model", default=None)
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--numref", type=int, help="Number of references", default=3)
    parser.add_argument("--inference_steps", type=int, help="Number of inference steps", default=50)
    parser.add_argument("--num_images_per_prompt", type=int, help="Number of generated images per prompt", default=4)
    parser.add_argument("--ip_adapter_scale", type=float, help="IP Adapter scale", default=0.6)
    parser.add_argument("--guidance_scale", type=float, help="Guidance scale", default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, help="Image guidance scale", default=8.)
    parser.add_argument("--adaptive_image_guidance_scale", type=float, help="Adaptive Image guidance scale", default=5.)

    args = parser.parse_args()
    return args


def main(args):
    print(args)
    sample(args.prompt,
           args.ref_images,
           args.ref_category,
           args.outdir,
           args.finetuned_path,
           args.num_images_per_prompt,
           args.inference_steps,
           args.numref,
           args.ip_adapter_scale,
           args.image_guidance_scale,
           args.adaptive_image_guidance_scale,
           args.guidance_scale,
           args.seed)


if __name__ == "__main__":
    args = parse_args()
    main(args)
