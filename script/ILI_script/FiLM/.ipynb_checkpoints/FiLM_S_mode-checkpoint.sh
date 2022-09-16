export CUDA_VISIBLE_DEVICES=5

for mode_type in 2 1 #192 720 #96 336 #720 #192 336 # 96 192 336 720

do

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_24_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 60 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab  2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 60 \
  --train_epochs 60 \
  --mode_type $mode_type \
  --ours
  
  

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_36_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 60 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab  2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 60 \
  --train_epochs 60 \
  --mode_type $mode_type \
  --ours

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_48_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 60 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab  2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 50 \
  --train_epochs 50 \
  --mode_type $mode_type \
  --ours
  

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_60_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 60 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab  2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 60 \
  --train_epochs 60 \
  --mode_type $mode_type \
  --ours

done
  