export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_720 \
  --model FiLM \
  --data ETTh2 \
  --features M \
  --seq_len  168 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 5 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 10 \
  --train_epochs 10 \
  --modes1 32 \
  --ab 2 --ours

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_96 \
  --model FiLM \
  --data ETTh2 \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 5 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 3 \
  --train_epochs 3 \
  --modes1 32 \
  --ab 2 --ours

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_192 \
  --model FiLM \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 5 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --modes1 32 \
  --patience 1 \
  --train_epochs 1 \
  --ab 2 --ours

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_336 \
  --model FiLM \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 5 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 1 \
  --train_epochs 1 \
  --modes1 32 \
  --ab 2 --ours


  #--add_noise_train 'True' \
 


  
