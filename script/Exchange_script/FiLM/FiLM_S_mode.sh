export CUDA_VISIBLE_DEVICES=4

for mode_type in 2 1 #192 720 #96 336 #720 #192 336 # 96 192 336 720

do

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 384 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --modes1 32 \
  --mode_type $mode_type \
  --train_epochs 20 \



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --modes1 32 \
  --mode_type $mode_type \
  --train_epochs 20 \

  
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --modes1 32 \
  --mode_type $mode_type \
  --train_epochs 20 \



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720_mode \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --modes1 32 \
  --mode_type $mode_type \
  --train_epochs 20 \

done