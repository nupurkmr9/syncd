import argparse
import json
import os

import numpy as np
import torch
from lightning.pytorch import seed_everything
from PIL import Image
from src.editor import AttentionStore, SharedAttnProc
from src.flux_syncd import FluxCustomPipeline
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
WIDTH = HEIGHT = 1024
NUM = 3
text_seq = 512
axes_dims_rope = (16, 56, 56)
torch_dtype = torch.bfloat16


def get_token_indices(prompts):
    token_indices = []
    ref_tokens = tokenizer.encode(prompts[0])[:-1]

    for prompt in prompts[1:]:
        start = 2
        end = 3
        tokens = tokenizer.encode(prompt)
        ref_tokens_length = len(ref_tokens)
        for i in range(len(tokens) - ref_tokens_length + 1):
            # Check if the sublist matches part of the larger list
            if tokens[i: i + ref_tokens_length] == ref_tokens:
                start = i
                end = i + ref_tokens_length
        token_indices.append([start, end])
    return token_indices


def run_dataset_gen(categories, outdir='./', inference_step=50, prompt_file=None, desc_prompt_file=None, save_attn_mask=False, seed=42, rank=0):

    torch.cuda.set_device(rank)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f'{outdir}/masks', exist_ok=True)

    pipe = FluxCustomPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', num=NUM, torch_dtype=torch_dtype).to("cuda")
    seed_everything(seed)

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

    metadata = []
    llama_gen_prompts = torch.load(prompt_file)
    llama_gen_desc_prompts = torch.load(desc_prompt_file)

    for _, cat_ in enumerate(categories):
        prompts = llama_gen_prompts[cat_]
        prompts = [[prompts[i+x] for x in range(NUM)] for i in np.arange(0, len(prompts)-NUM, NUM)]
        promptsdesc = llama_gen_desc_prompts[cat_]

        for numdesc, desc in enumerate(promptsdesc):
            for num_sample, prompt in enumerate(prompts):
                token_indices = get_token_indices([cat_] + prompt)

                editor = AttentionStore(token_indices=token_indices, num_att_layers=57, WIDTH=WIDTH)

                with torch.no_grad():
                    model_output_old = pipe(
                        [f'{x}. {desc}. wide angle shot' for x in prompt],
                        width=WIDTH,
                        height=HEIGHT,
                        num_inference_steps=inference_step,
                        joint_attention_kwargs={'editor': editor},
                        guidance_scale=3.5,
                        return_dict=False
                    )[0]
                    torch.cuda.empty_cache()

                attn_mask = editor.attention_maps.reshape(NUM, 1, WIDTH // 16, WIDTH // 16).float()

                for numref_ in range(len(model_output_old)):
                    im_ = model_output_old[numref_]
                    im_.save(f"{outdir}/{cat_}_{numdesc}_{num_sample}_{rank}_{numref_}.jpg")
                    if args.save_attn_mask:
                        im_ = Image.fromarray((torch.clip(attn_mask[numref_, 0], 0., 1.).cpu().numpy() * 255).astype(np.uint8))
                        im_.save(f"{outdir}/masks/{cat_}_{numdesc}_{num_sample}_{rank}_{numref_}.jpg")

                metadata.append({
                    'filenames': [f"{cat_}_{numdesc}_{num_sample}_{rank}_{numref_}.jpg" for numref_ in range(NUM)],
                    'prompts': prompt,
                    'objaverse_id': '',
                    'category': 'deformable',
                    })

            with open(f'{outdir}/metadata_{rank}.json', 'w') as f:
                json.dump(metadata, f)

    with open(f'{outdir}/metadata_{rank}.json', 'w') as f:
        json.dump(metadata, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Run a sampling scripts for the dreammatcher')
    parser.add_argument('--prompt_file', type=str, default='./assets/generated_prompts/prompts_deformable.pt')
    parser.add_argument('--desc_prompt_file', type=str, default='./assets/generated_prompts/prompts_desc_deformable.pt')
    parser.add_argument('--outdir', type=str, help='outdir', required=True)
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('--inference_step', type=int, help='seed', default=50)
    parser.add_argument('--filename', type=str, help='filename', default='assets/categories.txt')
    parser.add_argument('--save_attn_mask', action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    with open(args.filename, 'r') as f:
        categories = f.readlines()

    categories = [x.strip() for x in categories]

    run_dataset_gen(
                categories,
                prompt_file=args.prompt_file,
                desc_prompt_file=args.desc_prompt_file,
                save_attn_mask=args.save_attn_mask,
                seed=args.seed,
                outdir=args.outdir,
                inference_step=args.inference_step,
                rank=0
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
