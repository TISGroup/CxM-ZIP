# nohup bash train_tomo_00089_e_8178_deconv_conv.sh > train_tomo_00089_e_8178_deconv_conv.out &

device="0"

# data
energy="8178"
data_id="tomo_00089_e_${energy}"
data="/your_dataset/tomo_00089/AC3_C4p6_3DXANES/AC3_C4p6_3DXANES_TOMO-XANES.txt"

# model
model="ZIP_deconv_conv_1024"

cfg="./cfgs/${model}.yaml"

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
