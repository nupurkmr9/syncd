## Getting Started

We require a GPU with atleast 48GB VRAM. The base environment setup is described [here](https://github.com/nupurkmr9/syncd/blob/main/README.md#getting-started)

### Defomable dataset generation

```
cd dataset
python gen_deformable.py --save_attn_mask --outdir assets/metadata/deformable_data 
```

### Rigid object dataset generation

A sample dataset generation command on a single Objaverse asset: 

```
wget https://www.cs.cmu.edu/~syncd-project/assets/prompts_objaverse.pt -P assets/generated_prompts/
bash assets/unzip.sh assets/metadata/objaverse_rendering/

torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=12356  gen_rigid.py  --rootdir ./assets/metadata  --promptpath assets/generated_prompts/prompts_objaverse.pt  --outdir assets/metadata/rigid_data

```

<strong>Note:</strong> Different from the paper, we use [FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) here instead of [xflux](https://github.com/XLabs-AI/x-flux) based depth ControlNet model.


* Full Objaverse guided dataset generation (Coming soon)

### Generating prompts from LLM

Object and image background description for classes in `assets/categories.txt`:
```
python gen_prompts.py 
```


Background description for Objaverse assets:
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/Cap3D_automated_Objaverse_old.csv?download=true -O Cap3D_automated_Objaverse_old.csv
python gen_prompts.py --rigid --captions Cap3D_automated_Objaverse_old.csv
```