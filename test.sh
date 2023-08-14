export GLUE_DIR=/home/wzx/data/GLUE/GLUE-baselines/glue_data/
export BERT_PRETRAIN=/home/wzx/data/BERT_pretrained/uncased_L-12_H-768_A-12/
export SAVE_DIR=save/

# python classify.py \
#     --task mrpc \
#     --mode train \
#     --train_cfg config/train_mrpc.json \
#     --model_cfg config/bert_base.json \
#     --data_file $GLUE_DIR/MRPC/train.tsv \
#     --pretrain_file $BERT_PRETRAIN/bert_model.ckpt \
#     --vocab $BERT_PRETRAIN/vocab.txt \
#     --save_dir $SAVE_DIR \
#     --max_len 128 \
#     # --data_parallel 0 \
#     # --total_steps 10000


python classify.py \
    --task mrpc \
    --mode eval \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/test.tsv \
    --model_file $SAVE_DIR/model_steps_1530.pt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --max_len 128