export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=yelp
export TRAIN_FILE=/data/firewall_pt/data/adversary_training_corpora/fake/train_clean.csv
export TEST_FILE=/data/firewall_pt/data/adversary_training_corpora/fake/test_clean.csv
export OUTPUT_DIR=/data/tianyu/models

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name fake\
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --train_file $TRAIN_FILE \
  --validation_file $TEST_FILE \
  --output_dir $OUTPUT_DIR/$TASK_NAME/ \
  --logging_steps 10 \
  --save_steps 10000 \
  --overwrite_output_dir