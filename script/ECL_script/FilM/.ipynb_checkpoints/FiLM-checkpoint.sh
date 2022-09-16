export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model FiLM \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --train_epochs 20 \
  --modes1 32 \
  --batch_size 16 \
  --ab 2 --ours

 python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_192_192 \
  --model FiLM \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --train_epochs 20 \
  --modes1 32 \
  --batch_size 16 \
  --ab 2 --ours
  
  #--modes1 16

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_336_336 \
  --model FiLM \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --train_epochs 20 \
  --modes1 32 \
  --batch_size 16 \
  --ab 2 --ours
  
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_720_moe0 \
  --model FiLM \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 20 \
  --train_epochs 20 \
  --modes1 32 \
  --ab 2 --batch_size 16 --ours
  
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_1440_720 \
#   --model FiLM \
#   --data custom \
#   --features M \
#   --seq_len 1440 \
#   --label_len 0 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --lradj type4 \
#   --learning_rate 1e-3 \
#   --patience 20 \
#   --train_epochs 20 \
#   --modes1 32 \
#   --ab 2 --batch_size 2 --ours