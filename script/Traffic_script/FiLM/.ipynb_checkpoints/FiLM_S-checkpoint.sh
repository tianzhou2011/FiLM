export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 15 \
  --modes1 32 \
  --train_epochs 15 --ours


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 720 \
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
  --patience 15 \
  --train_epochs 15 --ours


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 15 \
  --batch_size 4 \
  --train_epochs 15 --ours


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model FiLM \
  --data custom \
  --features S \
  --seq_len 1440 \
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
  --patience 15 \
  --batch_size 2 \
  --train_epochs 15 --ours
