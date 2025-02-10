## Getting Started

The base environment setup is described [here](https://github.com/nupurkmr9/syncd/blob/main/README.md#getting-started)

### Sampling

```
cd method
mkdir pretrained_model
wget https://www.cs.cmu.edu/~syncd-project/assets/sdxl_finetuned_20k.ckpt -P pretrained_model
wget https://www.cs.cmu.edu/~syncd-project/assets/actionfigure_1.tar.gz
tar -xvzf actionfigure_1.tar.gz


python sample.py --prompt "an actionfigure riding a motorcycle" --ref_images actionfigure_1 --ref_category "actionfigure" --finetuned_path pretrained_model/sdxl_finetuned_20k.ckpt
```

### Training on our Dataset

* Download our dataset:

```
cd method
git lfs install
git clone https://huggingface.co/datasets/nupurkmr9/syncd
cd syncd
bash unzip.sh 
cd ..
```

* Train SDXL (requires 80GB VRAM for training at 1K resolution)
```
python main.py --base configs/train_sdxl.yaml --name sdxl_syncd 
```


### Training on your generated Dataset:

* Calculate DINOv2 and Aesthetics Score

```
python calculate_scores.py --batch_size 1 --folder <path-to-dataset>
```

* Model training

Update dataset hyperparameters, including path and filtering thresholds, at `data.params` in configs/train_sdxl.yaml

```
python main.py --base configs/train_sdxl.yaml

```