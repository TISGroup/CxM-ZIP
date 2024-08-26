# nohup bash train_tomo_00097_deconv_conv.sh > train_tomo_00097_deconv_conv.out &

device="0"

# data
data_id="tomo_00097"
data="/your_dataset/tomo_00097_00098/${data_id}.h5"

# model
model="ZIP_deconv_conv_1024"

cfg="./cfgs/${model}.yaml"

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
