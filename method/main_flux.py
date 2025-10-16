import argparse
import datetime
import os

import deepspeed
import torch
import torch.distributed as dist
import wandb
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from prodigyopt import Prodigy
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# local imports
from pipelines.flux_pipeline.model import SynCDDiffusion
from data.data import ConcatDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

def init_distributed():
    """Initialize distributed training"""
    deepspeed.init_distributed(
        dist_backend='nccl',
        auto_mpi_discovery=True,
        init_method='env://'
    )
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="flux_syncd", help="postfix for logdir")
    parser.add_argument("--logdir", type=str, default="logs", help="base folder for logs")
    parser.add_argument("--resume", type=str, help="resume from logdir or checkpoint")
    parser.add_argument("--base", default="configs/train_flux.yaml", help="path to base config")
    parser.add_argument("--seed", type=int, default=23, help="random seed")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mode", type=str, default="syncd", help="mode to train")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    return parser.parse_args()

def init_wandb(args, config):
    """Initialize wandb only on the main process"""
    if dist.get_rank() == 0:
        run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.name}"
        wandb.init(
            project=args.wandb_project,            
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True),
            settings=wandb.Settings(start_method="thread")
        )

def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.base)
    init_distributed()
    
    # Initialize wandb
    if args.wandb_project:
        init_wandb(args, config)
    
    # Setup logging directory
    dist.barrier()
    if not torch.distributed.is_initialized() or dist.get_rank() == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        logdir = os.path.join(args.logdir, f'{now}_{args.name}' if args.name else  f'{now}_flux')
        os.makedirs(logdir, exist_ok=True)
        # Save config
        OmegaConf.save(config, os.path.join(logdir, "config.yaml"))
    dist.barrier()


    # Initialize model
    model = SynCDDiffusion(
        pretrained_model_name_or_path=config.model.pretrained_model_name_or_path,
        regularization_prob=config.model.regularization_prob,
        num=config.model.num,
        add_lora=config.model.add_lora,
        rank=config.model.rank,
        trainkeys=config.model.trainkeys,
        masked=config.model.masked,
        shared_attn=config.model.shared_attn,
        **config.model.other_kwargs
    )
    if args.resume:
        model = load_state_dict_from_zero_checkpoint(model, args.resume, tag='')

    config.training.max_steps *= config.training.gradient_accumulation_steps

    # Update the ds_config
    ds_config = {
        "train_batch_size": config.data.params.batch_size * dist.get_world_size() * config.training.gradient_accumulation_steps,  # Global batch size
        "train_micro_batch_size_per_gpu": config.data.params.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "bf16": {
            "enabled": True  # Disable bf16 if using fp16
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": False
        },
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True
    }


    # Update model initialization
    optimizer = Prodigy(list(filter(lambda p: p.requires_grad, model.transformer.parameters())), 
                        lr=config.training.base_learning_rate,
                        weight_decay=1e-4,
                        decouple=True,
                        use_bias_correction=True,
                        safeguard_warmup=True,
                      )
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        model_parameters=list(filter(lambda p: p.requires_grad, model.transformer.parameters()))
    )
    
    # deepspeed.init_distributed()
    # Get data
    train_dataset = ConcatDataset(
        rootdir=config.data.params.rootdir,
        mode=config.data.params.mode,
        numref=config.data.params.numref,
        drop_im=config.data.params.drop_im,
        drop_txt=config.data.params.drop_txt,
        img_size=config.data.params.img_size,
        drop_both=config.data.params.drop_both,
        drop_mask=config.data.params.drop_mask,
        random_crop=config.data.params.random_crop,
        filter_dino=config.data.params.filter_dino,
        filter_aesthetics=config.data.params.filter_aesthetics,
        regularization=config.data.params.regularization,
    )
    # prepare dataloader
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False,
        seed=args.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=train_dataset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=config.data.params.num_workers,
    )
    # Training loop
    global_step = 0 
    while True:
        model_engine.train()
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()

            # Log metrics
            if batch_idx % config.training.log_every == 0:
                # Gather loss from all processes
                loss_tensor = torch.tensor(loss.item()).cuda()
                dist.all_reduce(loss_tensor)
                avg_loss = loss_tensor.item() / dist.get_world_size()
                
                if dist.get_rank() == 0 and args.wandb_project:  # Log only on main process
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_step": global_step,
                    }, step=global_step)
                    
                    print(f"step {global_step}, Step {batch_idx}, Loss: {avg_loss:.4f}")

            global_step += 1
            # Save checkpoint
            if global_step % config.training.save_every == 0:
                ckpt_path = f"step_{global_step}"
                # Save on all ranks since model is sharded
                dist.barrier()
                model_engine.save_checkpoint(logdir, ckpt_path)
                dist.barrier()
            if global_step > config.training.max_steps:
                break

        if global_step > config.training.max_steps:
            break
    
    # Make sure all processes are done before finishing
    dist.barrier()
    
    # Finish wandb run
    if args.wandb_project and dist.get_rank() == 0:
        wandb.finish()

if __name__ == "__main__":
    main() 