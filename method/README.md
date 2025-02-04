## Getting Started

Basic environment setup is descriped [here](https://github.com/nupurkmr9/syncd-project/blob/main/README.md#getting-started)

### Sampling

```
cd method
mkdir pretrained_model
wget https://www.cs.cmu.edu/~syncd-project/assets/sdxl_finetuned_20k.ckpt -P pretrained_model
wget https://www.cs.cmu.edu/~syncd-project/assets/actionfigure_1.tar.gz
tar -xvzf actionfigure_1.tar.gz


python sample.py --prompt "an actionfigure riding a motorcycle" --ref_images actionfigure_1 --ref_category "actionfigure" --finetuned_path pretrained_model/sdxl_finetuned_20k.ckpt
```

### Training on our Dataset (Coming Soon)



### Training on your generated Dataset:

* Calculate DINOv2 and Aesthetics Score

```
python calculate_scores.py --batch_size 1 --folder <path-to-dataset>
```

* Model training

We require a GPU with atleast 80GB VRAM for training at 1K resolution. 
Update path to the dataset at `data.params.rootdir` in configs/train_sdxl.yaml

```
python main.py --base configs/train_sdxl.yaml

```