# nohup bash train_tomo_00087_pixelshuffle.sh > train_tomo_00087_pixelshuffle.out &

device="0"

# data
data_id="tomo_00087"
data="/your_dataset/${data_id}/${data_id}.h5"

# model
model="ZIP_pixelshuffle_1024"

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
