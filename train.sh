export BERT_PRETRAIN=../data/BERT_pretrained/uncased_L-12_H-768_A-12/

python review.py \
    --mode train \
    --train_cfg config/train_tomato.json \
    --model_cfg config/bert_base.json \
    --pretrain_file $BERT_PRETRAIN/bert_model.ckpt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir save/ \
    --max_len 100 \
    # --data_parallel 0 \
    # --total_steps 10000


# python classify.py \
#     --task mrpc \
#     --mode eval \
#     --train_cfg config/train_mrpc.json \
#     --model_cfg config/bert_base.json \
#     --data_file $GLUE_DIR/MRPC/test.tsv \
#     --model_file $SAVE_DIR/model_steps_1530.pt \
#     --vocab $BERT_PRETRAIN/vocab.txt \
#     --max_len 128