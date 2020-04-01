#!/usr/bin/env bash
EXEC_ID=ImExplore-lam3e-8-expp0.25
DATA_DIR=~/data/wiki_book_32768
TOTAL_UPDATES=1000000
WARMUP_UPDATES=10000
PEAK_LR=0.0001
TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512
MAX_SENTENCES=8 # 32 for v100 and fp16
UPDATE_FREQ=8
SEED=100
LOG_DIR=~/exps/fairseq/${EXEC_ID}

echo 'Environment'
nvidia-smi
ls -alh
ls ~ -alh

echo 'Start Training'
python3 train.py --fp16 ${DATA_DIR} --ddp-backend=no_c10d \
    --distributed-no-spawn \
    --num-workers 4 --task electra --criterion electra_mask --loss-lamda 50.0 --mask-prob 0.15 \
    --embedding-normalize --generator-size-divider 3 \
    --arch electra_base_masker --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 10.0 \
    --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.01 \
    --max-sentences ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} --seed ${SEED} \
    --encoder-normalize-before  \
    --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 50 --tensorboard-logdir ~/${LOG_DIR} \
    --keep-updates-list 20000 50000 100000 200000 400000 600000 800000 1000000 \
    --save-interval-updates 10000 --keep-interval-updates 1 --no-epoch-checkpoints \
    --skip-invalid-size-inputs-valid-test --save-dir ~/exps/fairseq/${EXEC_ID}
