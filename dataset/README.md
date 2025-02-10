## Getting Started

We require a GPU with atleast 48GB VRAM. The base environment setup is described [here](https://github.com/nupurkmr9/syncd/blob/main/README.md#getting-started)

### Defomable dataset generation

```
cd dataset
python gen_deformable.py --save_attn_mask --outdir assets/metadata/deformable_data 
```

### Rigid dataset generation (sample)

A sample dataset generation command on a single Objaverse asset: 

```
wget https://www.cs.cmu.edu/~syncd-project/assets/prompts_objaverse.pt -P assets/generated_prompts/
bash assets/unzip.sh assets/metadata/objaverse_rendering/

torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=12356  gen_rigid.py  --rootdir ./assets/metadata  --promptpath assets/generated_prompts/prompts_objaverse.pt  --outdir assets/metadata/rigid_data

```

### Objaverse guided rigid dataset generation

<strong>Note:</strong> Different from the paper, we use [FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) here instead of [xflux](https://github.com/XLabs-AI/x-flux) for depth conditioning.

We used ~75000 [assets](assets/objaverse_ids.pt) from [Objaverse](https://objaverse.allenai.org). We re-rendered the assets again, following [Cap3D](https://huggingface.co/datasets/tiange/Cap3D) and provide them [here](https://huggingface.co/datasets/nupurkmr9/objaverse_rendering/tree/main). 

We first calculate multi-view correspondence, which requires installing `pytorch3D`. 

```
pip install objaverse
pip install ninja
pip install trimesh
pip install "git+https://github.com/facebookresearch/pytorch3d.git"  # or follow the steps [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source) for installation.

cd assets/metadata/objaverse_rendering
wget https://huggingface.co/datasets/nupurkmr9/objaverse_rendering/resolve/main/archive_1.zip  # a subset of objaverse renderings
unzip archive_1.zip
cd ../../../
bash assets/unzip.sh assets/metadata/objaverse_rendering/

python gen_corresp.py --download --rendered_path ./assets/metadata/objaverse_rendering --objaverse_path ./assets/metadata/objaverse_assets --outdir ./assets/metadata
```

Dataset generation Code:

```
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=12356  gen_rigid.py  --rootdir ./assets/metadata  --promptpath assets/generated_prompts/prompts_objaverse.pt  --outdir <output-path-to-save-dataset>

```


### Generating prompts from LLM for your own categories

Object and image background description for classes in `assets/categories.txt`:
```
python gen_prompts.py 
```


Background description for Objaverse assets:
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/Cap3D_automated_Objaverse_old.csv?download=true -O Cap3D_automated_Objaverse_old.csv
python gen_prompts.py --rigid --captions Cap3D_automated_Objaverse_old.csv
```