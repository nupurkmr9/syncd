## Getting Started

The base environment setup is described [here](https://github.com/nupurkmr9/syncd/blob/main/README.md#getting-started)

### Sampling

```
cd method
mkdir pretrained_model
wget https://www.cs.cmu.edu/~syncd-project/assets/sdxl_finetuned_20k.ckpt -P pretrained_model
wget https://huggingface.co/spaces/nupurkmr9/SynCD/resolve/main/models/pytorch_model.bin?download=true -O pretrained_model/pytorch_model.bin
wget https://www.cs.cmu.edu/~syncd-project/assets/actionfigure_1.tar.gz
tar -xvzf actionfigure_1.tar.gz

# sample from SDXL model
python sample.py --prompt "An action figure riding a motorcycle" --ref_images actionfigure_1 --ref_category "action figure" --finetuned_path pretrained_model/sdxl_finetuned_20k.ckpt

# sample from FLUX model
python sample_flux.py --prompt "An action figure on a beach. Waves in the background. Realistic shot." --ref_images actionfigure_1 --finetuned_path pretrained_model/pytorch_model.bin --numref 3
```

### Training on our Dataset

Training requires 80GB VRAM GPUs

* Download our dataset:

```
cd method
git lfs install
git clone https://huggingface.co/datasets/nupurkmr9/syncd
cd syncd
bash unzip.sh 
cd ..
```

* Train FLUX.
```
deepspeed main_flux.py --base configs/train_flux.yaml --name flux_syncd 
```


* Train SDXL
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