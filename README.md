# ZIP
optimiZed Implicit neural rePresentation for battery X-ray tomography

## Setup


### Package
The code is based on PyTorch and requires the following python packges.

```python
torch
torchvision
scikit-image
numpy
h5py
pytorch_msssim
yaml
```

### Dataset
The experimental dataset is available on [TomoBank](https://tomobank.readthedocs.io/en/latest/).

### Extract
Take [tomo_00098](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.nano.html) as an example. First, download the dataset, which is stored in h5 file. Then, extract the frames into image folders with [extract.py](..extract.py) and following arguments:
```bash
python3 extract.py --file ./data/tomo_00098.h5 --resolution 2048 --save_dir ./dataset/tomo_00098
```

### Training
```bash
device="0"

# data
dataset_name="tomo_00087"
dataset="./${dataset_name}"

# model
model="ZIP"

cfg="./cfgs/${model}.yaml"

print_freq="100"
save_freq="1"

log_dir="./runs/${dataset_name}"

CUDA_VISIBLE_DEVICES=$device python3 -u train.py \
  --cfg $cfg \
  --dataset $dataset \
  --log_dir $log_dir \
  --print_freq $print_freq \
  --save_freq $save_freq \
  --lr $lr \
  --epochs $epochs \
  --cuda
````


### Inference
```bash
device="0"

# log dir
dataset_name="tomo_00087"
log_dir="./runs/${dataset_name}"

CUDA_VISIBLE_DEVICES=$device python3 -u infer.py \
  --log_dir $log_dir \
  --cuda
````

### Generating h5 file
```bash
# log dir
dataset_name="tomo_00087"
log_dir="./runs/${dataset_name}"
save_dir="./generated/${dataset_name}"

CUDA_VISIBLE_DEVICES=$device python3 -u generate.py \
  --log_dir $log_dir \
  --save_dir $save_dir
````


### Citation
If this work is useful for your research, please kindly cite it:
```bibtex
@article{yan2024zip,
title={OptimiZed Implicit Neural Representation for Battery X-ray Tomography},
author={Yan, Zipei and Wang, Qiyu and Yu, Xiqian and Li, Jizhou and K.-P. Ng, Michael},
journal={Chinese Physics Letters},
year={2024}
```

### Contact
Please contact: lijz AT ieee DOT org




