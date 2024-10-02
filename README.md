# Compression of Battery X-ray Tomography Data with Machine Learning
Zipei Yan, Qiyu Wang, Xiqian Yu, Jizhou Li, and Michael K.-P. Ng. **_Chin. Phys. Lett._** 2024, 41 (9): 098901, DOI: [10.1088/0256-307X/41/9/098901](https://cpl.iphy.ac.cn/10.1088/0256-307X/41/9/098901).

## Setup

### Dependency
The code is based on PyTorch and requires the following python packges.

```text
h5py==3.10.0
numpy==2.1.0
opencv_python==4.9.0.80
pandas==2.2.2
pytorch_msssim==1.0.0
PyYAML==6.0.2
scipy==1.14.1
setuptools==59.6.0
skimage==0.0
tifffile==2024.2.12
torch==2.2.2
torchvision==0.17.2
tqdm==4.66.2
```

The above dependencies are listed in `requirements.txt`, you can install them with:
```bash
pip install -r requirements.txt
```

### Dataset
The data utilized in this work are from [TomoBank](https://tomobank.readthedocs.io/en/latest/). In our experiments, we utilize three data, including
[tomo_00087](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.nano.html#wedge), [tomo_00089](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.XANES.html#ssrl-xanes-tomography) and [tomo_00097](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.nano.html#absorption). And more details are reported in our manuscript Table 1. 

Note that [tomo_00089](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.XANES.html#ssrl-xanes-tomography) has multiple energies, therefore running experiments with it should specify the energy.


## Training

The training code is in `train.py`. And the detailed arguments could be found using `python3 train.py --help`, as shown below:
```shell
usage: train.py [-h] [--cfg CFG] [--data DATA] [--energy ENERGY] [--resolution RESOLUTION] [--cuda] [--log_dir PATH] [--lr LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--num_workers NUM_WORKERS] [--print_freq PRINT_FREQ] [--save_freq SAVE_FREQ] [--alpha ALPHA] [--beta BETA] [--gamma GAMMA] [--seed SEED]

ZIP Training script

options:
  -h, --help            show this help message and exit
  --cfg CFG             cfg file path (default: ./cfgs/ZIP_deconv_conv_1024.yaml), adapt to any resolution automatically, e.g., 64/128/256/512/1024, etc.
  --data DATA           h5 data path or txt path
  --energy ENERGY       energy for data with multiple energy points (default: None)
  --resolution RESOLUTION
                        resolution for training (default None), if None, use default resolution.
  --cuda                use cuda
  --log_dir PATH        where checkpoints and logs to be saved (default: ./runs)
  --lr LR               learning rate (default: 0.0005)
  --epochs EPOCHS       training epochs (default: 1000)
  --batch_size BATCH_SIZE
                        training batch size (default: 1)
  --num_workers NUM_WORKERS
                        number of workers for data loader (default: 4)
  --print_freq PRINT_FREQ
                        print frequency (default: 100)
  --save_freq SAVE_FREQ
                        save frequency for checkpoint (default: None), if None, save last only
  --alpha ALPHA         coefficient for L2
  --beta BETA           coefficient for SSIM
  --gamma GAMMA         coefficient for Laplacian
  --seed SEED           random seed (default: 42)
```

And for these three data, we have the following scrips to run the experiments, respectively. And these scrips are also included in the `scripts` folder.

### tomo_00087
```bash
device="0"

# data
data_id="tomo_00087"
data="/your_dataset/${data_id}/${data_id}.h5"

# model
model="ZIP_deconv_conv_1024"
cfg="./cfgs/${model}.yaml"

# config
print_freq="100"
save_freq="100"

lr="0.0005"
epochs="1000"
resolution="2048"

log_dir="./runs/${data_id}/${model},lr=${lr},epochs=${epochs},resolution=${resolution}"

CUDA_VISIBLE_DEVICES=$device python3 -u train.py \
  --cfg $cfg \
  --data $data \
  --log_dir $log_dir \
  --print_freq $print_freq \
  --save_freq $save_freq \
  --lr $lr \
  --epochs $epochs \
  --resolution $resolution \
  --cuda
````

### tomo_00089
Energy: 8178 ev

```bash
device="0"

# data
energy="8178"
data_id="tomo_00089_e_${energy}"
data="/your_dataset/tomo_00089/AC3_C4p6_3DXANES/AC3_C4p6_3DXANES_TOMO-XANES.txt"

# model
model="ZIP_deconv_conv_1024"
cfg="./cfgs/${model}.yaml"

# config
print_freq="100"
save_freq="100"

lr="0.0005"
epochs="1000"

log_dir="./runs/${data_id}/${model},lr=${lr},epochs=${epochs}"

CUDA_VISIBLE_DEVICES=$device python3 -u train.py \
  --cfg $cfg \
  --data $data \
  --log_dir $log_dir \
  --print_freq $print_freq \
  --save_freq $save_freq \
  --lr $lr \
  --epochs $epochs \
  --energy $energy \
  --cuda
```

Energy: 8570 eV
```bash
device="0"

# data
energy="8570"
data_id="tomo_00089_e_${energy}"
data="/your_dataset/tomo_00089/AC3_C4p6_3DXANES/AC3_C4p6_3DXANES_TOMO-XANES.txt"

# model
model="ZIP_deconv_conv_1024"

cfg="./cfgs/${model}.yaml"

# config
print_freq="100"
save_freq="100"

lr="0.0005"
epochs="1000"

log_dir="./runs/${data_id}/${model},lr=${lr},epochs=${epochs}"

CUDA_VISIBLE_DEVICES=$device python3 -u train.py \
  --cfg $cfg \
  --data $data \
  --log_dir $log_dir \
  --print_freq $print_freq \
  --save_freq $save_freq \
  --lr $lr \
  --epochs $epochs \
  --energy $energy \
  --cuda
```

### tomo_00097
```bash
device="0"

# data
data_id="tomo_00097"
data="/your_dataset/tomo_00097_00098/${data_id}.h5"

# model
model="ZIP_deconv_conv_1024"
cfg="./cfgs/${model}.yaml"

# config
print_freq="100"
save_freq="100"

lr="0.0005"
epochs="1000"
resolution='2048'

log_dir="./runs/${data_id}/${model},lr=${lr},epochs=${epochs},resolution=${resolution}"

CUDA_VISIBLE_DEVICES=$device python3 -u train.py \
  --cfg $cfg \
  --data $data \
  --log_dir $log_dir \
  --print_freq $print_freq \
  --save_freq $save_freq \
  --lr $lr \
  --epochs $epochs \
  --resolution $resolution \
  --cuda
````


## Inference

[//]: # (### tomo_00087 & tomo_00089 & tomo_00097)

Infer all saved checkpoints.
```bash
device="0"

log_dir="./runs/tomo_00097/ZIP_deconv_conv_1024,lr=0.0005,epochs=1000"

CUDA_VISIBLE_DEVICES=$device python3 -u infer.py \
  --log_dir $log_dir \
  --cuda
````

Infer checkpoints at epoch 900 and 1000.
```bash
device="0"

log_dir="./runs/tomo_00097/ZIP_deconv_conv_1024,lr=0.0005,epochs=1000"

CUDA_VISIBLE_DEVICES=$device python3 -u infer.py \
  --log_dir $log_dir \
  --epochs 900 1000 \
  --cuda
````


## Reconstruction

### tomo_00087 & tomo_00097
First generate `h5` file with the following script,
```bash
# log dir
log_dir="./runs/tomo_00087/ZIP_deconv_conv_1024,lr=0.0005,epochs=1000"
#save_dir="./generated/${dataset_name}"
epoch="1000"

eoch "the generated h5 file is in ${log_dir}/reconstructed"

CUDA_VISIBLE_DEVICES=$device python3 -u generate_h5file.py \
  --log_dir $log_dir \
  --epoch $epoch
````

Then, utilize [tomopy](https://tomopy.readthedocs.io/en/stable/) to perform a basic reconstruction.

The following code conducts the reconstruction for [tomo_00087](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.nano.html#wedge).
```bash
cd "./runs/tomo_00087/ZIP_deconv_conv_1024,lr=0.0005,epochs=1000"
tomopy recon --file-name tomo_00087.h5 --rotation-axis 1196.0
```

The following code conducts the reconstruction for [tomo_00097](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.nano.html#absorption).
```bash
cd "./runs/tomo_00097/ZIP_deconv_conv_1024,lr=0.0005,epochs=1000"
tomopy recon --file-name tomo_00097.h5 --rotation-axis 1254.0
```

### tomo_00089

For [tomo_00089](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.XANES.html#ssrl-xanes-tomography) with enery equals to 8178, its reconstruction is performed as:
```python
import os
import numpy as np
import tomopy
import skimage

# Set the input and output directories
data_dir = './runs/tomo_00089_e_8178/ZIP_deconv_conv_1024,lr=0.0005,epochs=1000/generated/epoch1000'
save_dir = './runs/tomo_00089_e_8178/ZIP_deconv_conv_1024,lr=0.0005,epochs=1000/reconstructed/epoch1000'

# Create the save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the projection data
proj_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
proj = np.stack([skimage.io.imread(os.path.join(data_dir, f)) for f in proj_files], axis=0)

# Perform the reconstruction using ASTRA SIRT-CUDA
angles = tomopy.angles(180)
rec = tomopy.recon(proj, angles, algorithm=tomopy.astra,
options={'method': 'SIRT_CUDA', 'num_iter': 200, 'proj_type': 'cuda', 'extra_options': {'MinConstraint': 0}})

# rec = tomopy.recon(proj, angles, algorithm='gridrec')

# Save the reconstruction as an image series
for i, img in enumerate(rec):
    # tiffile.imsave(os.path.join(save_dir, f'slice_{i:04d}.tiff'), img)
    i_str = str(i).zfill(len(str(rec)))
    skimage.io.imsave(os.path.join(save_dir, f'{i_str}.tif'), img, check_contrast=False)
```


## Citation
If this work is useful for your research, please kindly cite it:
```bibtex
@article{yan2024zip,
	author={Yan, Zipei and Wang, Qiyu and Yu, Xiqian and Li, Jizhou and Ng, Michael K. -P.},
	title={Compression of Battery X-ray Tomography Data with Machine Learning},
	journal={Chinese Physics Letters},
	volume={41},
	number={9},
	url={http://iopscience.iop.org/article/10.1088/0256-307X/41/9/098901},
	year={2024}
}
```

## Contact
Please contact: lijz AT ieee DOT org




