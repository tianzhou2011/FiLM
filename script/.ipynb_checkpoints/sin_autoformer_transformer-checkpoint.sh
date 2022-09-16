export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sin/ \
  --data_path sin.csv \
  --model_id ECL_96_96 \
  --model Transformer \
  --data sin \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1

