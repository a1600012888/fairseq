#!/usr/bin/env bash

EXEC_ID=robert-base-0.2
DATA_DIR=~/data/wiki_book_32768
TOTAL_UPDATES=1000000
WARMUP_UPDATES=10000
PEAK_LR=0.0001
TOKENS_PER_SAMPLE=128
MAX_POSITIONS=128
MAX_SENTENCES=32 # 32 for v100 and fp16
UPDATE_FREQ=2
SEED=100
LOG_DIR=~/exps/fairseq/deepsup-${EXEC_ID}

echo 'Environment'
nvidia-smi
ls -alh
ls ~ -alh


echo 'Start Training'
python3 train.py ${DATA_DIR} --ddp-backend=no_c10d  --restore-file=/home/tianyuan/exps/fairseq/deepsup-robert-base-0.2/checkpoint_last.pt \
    --task masked_lm --criterion masked_lm_deepsup --deepsup_lambda 0.2 \
    --arch roberta_base_deepsup --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.01 \
    --max-sentences ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} --seed ${SEED} \
    --encoder-normalize-before  \
    --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 100 --tensorboard-logdir ~/${LOG_DIR} \
    --save-interval-updates 25000 --keep-interval-updates 5 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test --save-dir ~/exps/fairseq/deepsup-${EXEC_ID}
