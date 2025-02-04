import argparse
import datetime
import glob
import os
import sys
from contextlib import contextmanager
from typing import Generator

import lightning as L
import numpy as np
import torch
import torchvision
import wandb
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities import rank_zero_only
from natsort import natsorted
from omegaconf import OmegaConf
from PIL import Image
from pipelines.util import instantiate_from_config


class CustomDeepSpeedStrategy(DeepSpeedStrategy):
    @contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        # lightning DeepSpeedStrategy used to call deepspeed.zero.Init() here
        yield


deepspeed_config = {
    "zero_allow_untested_optimizer": True,
    "bf16": {
        "enabled": True
    },
    "train_micro_batch_size_per_gpu":1, 
    "zero_optimization": {
        "stage": 3,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
}


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
MULTINODE_HACKS = True


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--name", type=str, const=True, default="", nargs="?", help="postfix for logdir",)
    parser.add_argument("--resume", type=str, default="", nargs="?", help="resume from logdir or checkpoint in logdir",)
    parser.add_argument("--resume_from_checkpoint", type=str, default="", nargs="?", help="resume from specific checkpoint",)
    parser.add_argument("--base", help="paths to base config. Loaded from left-to-right. ", default="configs/train_sdxl.yaml")
    parser.add_argument("--seed", type=int, default=23, help="seed for seed_everything",)
    parser.add_argument("--projectname", type=str, default="gencd",)
    parser.add_argument("--logdir", type=str, default="logs", help="directory for logging dat shit",)
    parser.add_argument('--scale_lr', action="store_true", help="scale base-lr by ngpu * batch_size * n_accumulate",)
    parser.add_argument("--wandb", action="store_true", help="log to wandb")
    args = parser.parse_args()
    return args


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "*.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print('available "last" checkpoints:')
    print(ckpt)
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
    else:
        ckpt = ckpt[0]
    print(f"Most recent ckpt is {ckpt}")
    return ckpt


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, ckpt_name=None,):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: L.Trainer, pl_module, exception):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            if self.ckpt_name is None:
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            else:
                ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time
                time.sleep(5)
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),)

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),)
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, disabled=False, log_on_batch_idx=False, log_before_first_step=False, enable_autocast=True, log_images_kwargs=None):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_before_first_step = log_before_first_step
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0) and (split=='val' or self.check_frequency(check_idx)):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.amp.autocast('cuda', **gpu_autocast_kwargs):
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            torch.cuda.empty_cache()

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

            root = os.path.join(pl_module.logger.save_dir, "images", split)
            os.makedirs(root, exist_ok=True)
            for k in images:
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, pl_module.global_step, pl_module.current_epoch, batch_idx)
                path = os.path.join(root, filename)
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                img = Image.fromarray(grid)
                img.save(path)
            
                if pl_module is not None and isinstance(pl_module.logger, WandbLogger):
                    pl_module.logger.log_image(key=f"{split}/{k}", images=[img,], step=pl_module.global_step,)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq) == 0:
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.global_rank == 0 and 'all' not in pl_module.trainkeys:
            # print("on save checkpoint callback", pl_module.trainkeys)
            st = checkpoint["state_dict"]
            layers = []
            for key in list(st.keys()):
                for keytosave in pl_module.trainkeys:
                    if keytosave in key:
                        layers.append(key)
            checkpoint['delta_state_dict'] = {}
            for each in layers:
                checkpoint['delta_state_dict'][each] = st[each]
            del checkpoint['state_dict']
            checkpoint['state_dict'] = checkpoint['delta_state_dict']


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, period: int, **kwargs):
        super().__init__(**kwargs)
        self.period = period

    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs):
        if pl_module.global_step % self.period == 0 and pl_module.global_step > 0:
            filename = f'{pl_module.global_step}.ckpt'
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs):
        pass

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs):
        pass


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    opt = parse_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = get_checkpoint_name(logdir)

        print(f'Resuming from checkpoint "{ckpt}"')

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        nowname =now + ("_" + opt.name if opt.name else "")
        logdir = os.path.join(opt.logdir, nowname)
        print(f"LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    try:
        # init and save configs
        config = OmegaConf.load(opt.base)
        lightning_config = config.pop("lightning", OmegaConf.create())

        gpuinfo = lightning_config["trainer"]["devices"]
        print(f"Running on GPUs {gpuinfo}")

        # trainer and callbacks
        trainer_kwargs = dict()
        default_logger_cfgs = {
            "wandb": {
                "target": "lightning.pytorch.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "id": nowname,
                    "project": opt.projectname,
                    "log_model": False,
                },
            },
            "csv": {
                "target": "lightning.pytorch.loggers.CSVLogger",
                "params": {
                    "name": "testtube",  # hack for sbord fanatics
                    "save_dir": logdir,
                },
            },
        }
        default_modelckpt_cfg = {
            "target": "lightning.pytorch.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}-{step}",
                "verbose": True,
                "save_top_k": -1,
                "every_n_train_steps": 2000,
            },
        }
        default_strategy_config = {
            "target": "lightning.pytorch.strategies.DDPStrategy",
            "params": {
                "find_unused_parameters": False
            }
        }
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                },
            },
            "learning_rate_logger": {
                "target": "lightning.pytorch.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                },
            }
        }

        logger_cfg = OmegaConf.merge(default_logger_cfgs["wandb" if opt.wandb else "csv"], OmegaConf.create())
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, lightning_config.modelcheckpoint)
        strategy_cfg = OmegaConf.merge(default_strategy_config, OmegaConf.create())
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, lightning_config.callbacks)

        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["plugins"] = list()

        trainer_kwargs = {key: val for key, val in trainer_kwargs.items() if key not in lightning_config["trainer"]}
        if 'flux' in config.model.target:
            del trainer_kwargs['strategy']
            trainer = Trainer(strategy=CustomDeepSpeedStrategy(config=deepspeed_config), **lightning_config["trainer"], **trainer_kwargs)
        else:
            trainer = Trainer(**lightning_config["trainer"], **trainer_kwargs)

        # data and model
        model = instantiate_from_config(config.model)
        data = instantiate_from_config(config.data)
        data.prepare_data()
        print("#### Data #####")
        try:
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            print("datasets not yet initialized.")

        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
        accumulate_grad_batches = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        import signal
        signal.signal(signal.SIGUSR1, melk)

        # run
        print(f"starting train")
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    except Exception:
        raise
    finally:
        if opt.wandb:
            wandb.finish()
