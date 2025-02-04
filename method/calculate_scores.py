
import argparse
import os
import sys
from os.path import expanduser
from pathlib import Path
from urllib.request import urlretrieve

import clip
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torchvision import transforms
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor

sys.path.append('../method')
from data.data import CustomLoader

NUM = 3
def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def calc_scores(DIRS, batch_size, mode='rigid', rank=0):
    torch.cuda.set_device(rank)
    device = 'cuda'

    model = get_aesthetic_model()
    model = model.to(device)
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').eval().to(device)
    
    preprocess = transforms.Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    processor_dino = transforms.Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    for DIR in DIRS:
        data = CustomLoader(batch_size=batch_size, num_workers=4, transform=preprocess, filter_dino=-1., filter_aesthetics=-1., mode=mode, numref=-1, cropped_image=False, rootdir=[[DIR]])
        loader = data.train_dataloader()

        data_dino = CustomLoader(batch_size=batch_size, num_workers=4, transform=processor_dino, cropped_image=True, mode=mode, numref=-1,  filter_dino=-1., filter_aesthetics=-1., rootdir=[[DIR]])
        loader_dino = data_dino.train_dataloader()

        path = str(Path(DIR).stem)
        outdir = str(Path(DIR).parent)
        aesthetics_scores = {}
        dino_scores = {}

        for i, (batch, batch_dino) in enumerate(zip(loader, loader_dino)):
            batch_size = max(1, batch['images'].shape[0] // NUM)
            with torch.no_grad():
                emb = clip_model.encode_image(batch['images'].to(device)).float()
                emb /= emb.norm(dim=-1, keepdim=True)
                scores = model(emb)
            
            with torch.no_grad():
                image_embs = model_dino(batch_dino['images'].to(device))
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                image_embs = rearrange(image_embs, "(b n) ...-> b n ...", b=batch_size)
                scores_dino = torch.einsum("bni,bim->bnm", image_embs, image_embs.permute(0,2,1))

            for k in range(scores.shape[0]):
                aesthetics_scores[str(Path(batch['filenames'][k]).stem)] = scores[k][0].item()
            
            for k in range(scores_dino.shape[0]):
                score = torch.triu(scores_dino[k], diagonal=1)
                key = '+'.join([Path(x).stem for x in batch_dino['filenames']]) 
                dino_scores[key] = score.cpu().numpy()

        torch.save(aesthetics_scores, f'{outdir}/{path}_aesthetics.pt')
        torch.save(dino_scores, f'{outdir}/{path}_dino.pt')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a sampling scripts for the multi-view editing"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, default='deformable')
    parser.add_argument("--folder", type=str, required=True)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    gpus = range(8)
    args.folder = args.folder.split('+')
    print(args.folder)

    for DIR in args.folder:
        if os.path.isdir(DIR):
            print(DIR)
            calc_scores([DIR],
                        args.batch_size,
                        args.mode,
                        0
                )
        