## Download SynCD

You can download our filtered generated dataset [here](https://huggingface.co/datasets/nupurkmr9/syncd/tree/main) and start model training as detailed [here](https://github.com/nupurkmr9/syncd/blob/main/README.md#training-on-our-dataset)

## Getting Started (to generate your own dataset)

We require a GPU with atleast 48GB VRAM. 
The base environment setup. Same as [here](https://github.com/nupurkmr9/syncd/blob/main/README.md#getting-started)

```
git clone https://github.com/nupurkmr9/syncd.git
cd syncd
conda create -n syncd python=3.10
conda activate syncd
pip3 install torch torchvision torchaudio  # (Or appropriate torch>2.0 from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
pip install -r assets/requirements.txt
```

### Defomable dataset generation

```
cd dataset
python gen_deformable.py --save_attn_mask --outdir assets/metadata/deformable_data 
```

### Rigid dataset generation 

**Sample dataset generation** on a single Objaverse asset:

```
wget https://www.cs.cmu.edu/~syncd-project/assets/prompts_objaverse.pt -P assets/generated_prompts/
bash assets/unzip.sh assets/metadata/objaverse_rendering/

torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=12356  gen_rigid.py  --rootdir ./assets/metadata  --promptpath assets/generated_prompts/prompts_objaverse.pt  --outdir assets/metadata/rigid_data

```

**Full Objaverse guided dataset generation**

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

Dataset generation:

```
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=12356  gen_rigid.py  --rootdir ./assets/metadata  --promptpath assets/generated_prompts/prompts_objaverse.pt  --outdir <dir-path-to-save-dataset>

```

### Calculating DINO and Aesthetic Score

```python

python cal_scores.py --rootdir <dir-path-of-saved-dataset>

```


### Generating your own prompts given category names or 3D asset descriptions

Object and image background description for classes in `assets/categories.txt`:
```
python gen_prompts.py 
```


Background description for Objaverse assets:
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/Cap3D_automated_Objaverse_old.csv?download=true -O Cap3D_automated_Objaverse_old.csv
python gen_prompts.py --rigid --captions Cap3D_automated_Objaverse_old.csv
```