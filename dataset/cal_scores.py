
import argparse
import json
import os
import sys
from os.path import expanduser
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from torchvision import transforms
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor
from tqdm import tqdm

import clip

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


def calc_scores(rootdir):
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

    mean_dino_score = 0
    mean_aesthetic_score = 0
    batch_size = 1

    data = CustomLoader(batch_size=batch_size, num_workers=4, transform=preprocess, filter_dino=-1., filter_aesthetics=-1., numref=-1, cropped_image=False, rootdir=[[rootdir]], train=False, shuffle=False)
    if len(data.train_dataset.dataset[0].metadata) == 0:
        return

    loader = data.train_dataloader()

    data_dino = CustomLoader(batch_size=batch_size, num_workers=4, transform=processor_dino, numref=-1,  filter_dino=-1., filter_aesthetics=-1., cropped_image=True, rootdir=[[rootdir]], train=False, shuffle=False)
    loader_dino = data_dino.train_dataloader()

    prev_metadata_json = data.train_dataset.dataset[0].metadata_json
    metadata_json = []

    num_samples = 0
    num_images = 0
    for i, (batch, batch_dino) in enumerate(tqdm(zip(loader, loader_dino))):
        metadata_json.append(prev_metadata_json[i])
        with torch.no_grad():
            emb = clip_model.encode_image(batch['images'].to(device)).float()
            emb /= emb.norm(dim=-1, keepdim=True)
            scores = model(emb)
        
        with torch.no_grad():
            image_embs = model_dino(batch_dino['images'].to(device))
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            image_embs = rearrange(image_embs, "(b n) ...-> b n ...", b=batch_size)
            scores_dino = torch.einsum("bni,bim->bnm", image_embs, image_embs.permute(0,2,1))

        metadata_json[-1]['aesthetics_scores'] = {}
        for k in range(scores.shape[0]):
            metadata_json[-1]['aesthetics_scores'][str(Path(batch['filenames'][k]).stem)] = scores[k][0].item()
            mean_aesthetic_score += scores[k][0].item()
            num_images += 1
        
        for k in range(scores_dino.shape[0]):
            score = torch.triu(scores_dino[k], diagonal=1)
            metadata_json[-1]['dino_scores'] = score.cpu().numpy().tolist()
            mean_dino_score += score.cpu().numpy()[np.triu_indices(score.shape[0], k=1)].mean() 
            num_samples += 1

        
        if i % 100 == 0:
            print(i/100,mean_aesthetic_score / (num_images + 1), mean_dino_score / (num_samples + 1))
            
    print(mean_aesthetic_score / (num_images + 1), mean_dino_score / (num_samples + 1))

    with open(f'{rootdir}/metadata_with_scores.json', 'w') as f:
        json.dump(metadata_json, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Run a sampling scripts for the multi-view editing")
    parser.add_argument("--rootdir", type=str, required=True) 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    calc_scores(args.rootdir)
        