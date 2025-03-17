import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from peft import LoraConfig
from PIL import Image

sys.path.append('./')
from data.utils_ import square_crop_shortest_side
from pipelines.flux_pipeline.pipeline import SynCDFluxPipeline
from pipelines.flux_pipeline.transformer import FluxTransformer2DModelWithMasking


@torch.no_grad()
def decode(latents, pipeline):
    latents = latents / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    return image


@torch.no_grad()
def encode_target_images(images, pipeline):
    latents = pipeline.vae.encode(images).latent_dist.sample()
    latents = latents * pipeline.vae.config.scaling_factor
    return latents


def sample(prompt, ref_images, outdir, finetuned_path, num_images_per_prompt, inference_steps, numref, guidance_scale, true_cfg_scale, image_guidance_scale, seed):

    torch_dtype = torch.bfloat16
    device='cuda'
    height = 512
        
    transformer = FluxTransformer2DModelWithMasking.from_pretrained(
                'black-forest-labs/FLUX.1-dev',
                subfolder='transformer',
                torch_dtype=torch_dtype
            )
    pipeline = SynCDFluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', transformer=transformer, torch_dtype=torch_dtype)
    for name, attn_proc in pipeline.transformer.attn_processors.items():
        attn_proc.name = name

    target_modules=[
                    "to_k",
                    "to_q",
                    "to_v",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "proj_mlp",
                    "proj_out",
                    ]
    lora_rank = 32
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipeline.transformer.add_adapter(lora_config)
    if finetuned_path is not None:
        finetuned_path = torch.load(finetuned_path, map_location='cpu')
        transformer_dict = {}
        for key,value in finetuned_path.items():
            if 'transformer.base_model.model.' in key:
                transformer_dict[key.replace('transformer.base_model.model.', '')] = value 
        pipeline.transformer.load_state_dict(transformer_dict, strict=False)
    
    pipeline.to(device)
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()


    cat_ = Path(ref_images).stem
    os.makedirs(f'{outdir}/{cat_}', exist_ok=True)

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".JPG"])

    image_paths = [x for x in glob.glob(f'{ref_images}/*') if is_image_file(x)][:numref]
    images = torch.cat([2. * torch.from_numpy(np.array(square_crop_shortest_side(Image.open(img).convert('RGB')).resize((512, 512)))).permute(2, 0, 1).unsqueeze(0).to(torch_dtype)/255. -1. for img in image_paths])
    images = images.to(pipeline.device)

    counter = 0
    numref += 1
    for _ in range(num_images_per_prompt):
        latents = encode_target_images(images, pipeline)
        latents = torch.cat([torch.zeros_like(latents[:1]), latents], dim=0)
        masklatent = torch.zeros_like(latents)
        masklatent[:1] = 1.
        latents = rearrange(latents, "(b n) c h w -> b c h (n w)", n=numref)
        masklatent = rearrange(masklatent, "(b n) c h w -> b c h (n w)", n=numref)
        B, C, H, W = latents.shape
        latents = pipeline._pack_latents(latents, B, C, H, W)
        masklatent = pipeline._pack_latents(masklatent.expand(-1, C, -1, -1) ,B, C, H, W)

        generated_image = pipeline(prompt,
                latents_ref=latents,
                latents_mask=masklatent,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
                height=height,
                width=numref * height,
                generator=torch.Generator(device="cuda").manual_seed(seed + counter),
                joint_attention_kwargs={'shared_attn': True, 'num': numref},
                return_dict=False,
                negative_prompt="3d render, cartoon, low resolution, illustration, blurry, unrealistic",
                true_cfg_scale=true_cfg_scale,
                image_cfg_scale=image_guidance_scale,
            )[0][0]
        torch.cuda.empty_cache()
        generated_image = rearrange(generated_image, "b c h (n w) -> (b n) c h w", n=numref)
        for _, img in enumerate(generated_image[::numref]):
            img = Image.fromarray(((torch.clip(img.float(), -1., 1.).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8))
            name = f'{outdir}/{cat_}/{counter:05d}.png'
            img.save(name)
            counter += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Run the sampling script")
    parser.add_argument("--prompt", type=str, help="prompt", default="An actionfigure wearing a hat")
    parser.add_argument("--ref_images", type=str, help="path to reference images")
    parser.add_argument("--outdir", type=str, default="./samples")
    parser.add_argument("--finetuned_path", type=str, help="path to finetuned model", default=None)
    parser.add_argument("--num_images_per_prompt", type=int, help="Number of generated images per prompt", default=4)
    parser.add_argument("--inference_steps", type=int, help="Number of inference steps", default=30)
    parser.add_argument("--numref", type=int, help="Number of references", default=1)
    parser.add_argument("--guidance_scale", type=float, help="Guidance scale", default=3.5)
    parser.add_argument("--true_cfg_scale", type=float, help="Guidance scale for real images", default=1.0)
    parser.add_argument("--image_guidance_scale", type=float, help="Image guidance scale", default=1.0)
    parser.add_argument("--seed", type=int, help="seed", default=42)
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    sample(args.prompt,
            args.ref_images,
            args.outdir,
            args.finetuned_path,
            args.num_images_per_prompt, 
            args.inference_steps, 
            args.numref,
            args.guidance_scale,
            args.true_cfg_scale,
            args.image_guidance_scale,
            args.seed,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)

