import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from pytorch_lightning import seed_everything
from src.dataloader import ObjaverseDataset
from src.editor import SharedAttnProc
from src.flux_depth_syncd import FluxControlCustomPipeline
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

WIDTH = HEIGHT = 512
NUM = 3
text_seq = 512
axes_dims_rope = (16, 56, 56)
torch_dtype = torch.bfloat16


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def run_dataset_gen(categories, outdir='./', inference_step=50, rootdir='./', guidance_scale=10., warp_thresh=0.8, negative_prompt=None, warping=True, seed=42, rank=0, device='cuda', promptpath=None,):
    
    pipe = FluxControlCustomPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", num=NUM, torch_dtype=torch_dtype).to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    
    seed_everything(seed)

    dataset = ObjaverseDataset(categories, rootdir, promptpath, warping=warping, img_size=WIDTH)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=3,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=False
    )

    attn_procs = dict()
    for name, attn_proc in pipe.transformer.attn_processors.items():
        if name.startswith("single_transformer_blocks"):
            attn_procs[name] = SharedAttnProc(
                attn_proc, selfattn=True, single_transformer=True, NUM=NUM
            )
        else:
            attn_procs[name] = SharedAttnProc(
                attn_proc, selfattn=True, NUM=NUM
            )

    pipe.transformer.set_attn_processor(attn_procs)

    num_sample = 0
    metadata = []
    for _, batch in enumerate(tqdm(loader)):
        if batch is None:
            continue
        images = batch['images']
        prompts = batch['prompts']
        mask_images = batch['masks'].to(torch_dtype)
        depth_images = batch['depths'].to(device).to(torch_dtype)
        correspondence = batch['correspondence'].to(torch_dtype).to(device).to(torch_dtype) if warping else None
        counter_cc = batch['counter_cc'].to(device).long() if warping else None

        hw = (WIDTH // 16) * (WIDTH // 16)
        kernel_tensor = torch.ones((1, 1, 9, 9)).to(device).to(torch_dtype)
        attn_mask = F.interpolate(  (mask_images.unsqueeze(1) > 0)*1., (WIDTH // 16, WIDTH // 16), mode='area').to(device).to(torch_dtype)
        attn_mask = (torch.clamp(torch.nn.functional.conv2d(attn_mask, kernel_tensor, padding='same'), 0, 1) > 0)*1.

        attn_mask = torch.cat([rearrange(attn_mask, "(b n) c h w -> b 1 n c h w", n=NUM)] + 
                                [torch.roll(rearrange(attn_mask, "(b n) c h w -> b 1 n c h w", n=NUM), shifts=i, dims=2) for i in range(1,NUM)], dim=1)
        attn_mask = rearrange(
            attn_mask, "b n1 n c h w-> (b n) (n1 h w) c"
        )
        attn_mask = torch.cat([torch.ones_like(attn_mask[:, :text_seq, :]), attn_mask], 1)
        attn_mask = torch.einsum("b i d, b j d -> b i j", torch.ones_like(attn_mask[:, :text_seq + hw]), attn_mask)
        attn_mask[:, :text_seq + hw, :text_seq + hw]=1
        attn_mask[:, :text_seq, text_seq + hw:]=0
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -65504.0)
        attn_mask = rearrange(attn_mask.unsqueeze(0).expand(24, -1, -1, -1), "nh b ... -> b nh ...")

        model_output = pipe(
            prompt=[f'{x} a natural image with the object on the ground.' for x in prompts[:NUM]], 
            control_image=depth_images,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=inference_step, 
            joint_attention_kwargs={'attention_mask': attn_mask},
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            correspondence=correspondence, 
            counter_cc=counter_cc,
            warp_thresh=warp_thresh,
            negative_prompt=negative_prompt,
        ).images.cpu()
        torch.cuda.empty_cache()
        attn_mask = None

        for num_pairs in range(len(model_output) // NUM ):
            for numref_ in range(NUM):
                im_ = Image.fromarray(((torch.clip(model_output[(NUM)*num_pairs + numref_].permute(1, 2, 0) * 0.5 + 0.5, 0., 1.0)).float().cpu().numpy() * 255).astype(np.uint8))
                im_.save(f"{outdir}/{num_sample}_{rank}_{numref_}.jpg")
                im_ = Image.fromarray(((torch.clip(images[(NUM)*num_pairs + numref_].permute(1, 2, 0) * 0.5 + 0.5, 0., 1.0)).float().cpu().numpy() * 255).astype(np.uint8))
                im_.save(f"{outdir}/masks/{num_sample}_{rank}_{numref_}.jpg")
                
            metadata.append({
                'filenames': [f"{outdir}/{num_sample}_{rank}_{numref_}.jpg" for numref_ in range(NUM)],
                'prompts': prompts,
                'original_filenames': batch['imagenames'],
                'objaverse_id': str(Path(batch['imagenames'][0]).parent.stem),
                'category': 'rigid',
                })
            num_sample += 1

        if num_sample % 50 == 0:
            with open(f'{outdir}/metadata_{rank}.json', 'w') as f:
                json.dump(metadata, f)
    
    with open(f'{outdir}/metadata_{rank}.json', 'w') as f:
        json.dump(metadata, f)
                        

def parse_args():
    parser = argparse.ArgumentParser(description='Run a sampling scripts for the dreammatcher')
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('--inference_step', type=int, help='seed', default=30)
    parser.add_argument('--outdir', type=str, help='outdir', required=True)
    parser.add_argument('--rootdir', type=str, help='outdir', required=True)
    parser.add_argument('--promptpath', type=str, help='outdir', required=True)
    parser.add_argument('--guidance_scale', type=float, help='seed', default=10.0)
    parser.add_argument('--warp_thresh', type=float, help='seed', default=.8)
    parser.add_argument('--negative_prompt', type=str, default='3d render, cartoon, low resolution, illustration, blurry')
    parser.add_argument('--skip_warping', action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    seed_everything(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f'{args.outdir}/masks', exist_ok=True)
    categories =  list(torch.load('assets/objaverse_ids.pt'))
    categories.sort()
    categories = list(set(categories).intersection(set([str(Path(x).stem) for x in glob.glob(f'{args.rootdir}/objaverse_rendering/*.zip')])))

    run_dataset_gen(
                categories,
                outdir=args.outdir,
                inference_step=args.inference_step,
                rootdir=args.rootdir,
                guidance_scale=args.guidance_scale,
                warp_thresh=args.warp_thresh,
                promptpath=args.promptpath,
                warping=not args.skip_warping,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                rank=rank,
                device=device,
                )

    print("Done!")


if __name__ == "__main__":
    # distributed setting
    args = parse_args()
    main(args)
