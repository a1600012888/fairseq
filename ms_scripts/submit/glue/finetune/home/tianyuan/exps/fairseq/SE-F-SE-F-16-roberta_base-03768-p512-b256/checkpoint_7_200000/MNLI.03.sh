
PROBLEM=MNLI
METRIC=accuracy
CHECKPOINT_FILE=checkpoint_7_200000
PROCEDURE_FOLDER=/home/tianyuan/exps/fairseq/SE-F-SE-F-16-roberta_base-03768-p512-b256/
BERT_MODEL_PATH=/home/tianyuan/exps/fairseq/SE-F-SE-F-16-roberta_base-03768-p512-b256//checkpoint_7_200000.pt

N_CLASSES=3
ARCH=roberta_learner
N_SENT=392702
CRITERION=cross_entropy_classify
REGRESSION=
SEED_LIST="100 200 300" 
N_EPOCH_LIST="10" 
BATCH_SZ_LIST="16 32" 
LR_LIST="0.00004" 
WEIGHT_DECAY_LIST="0.1" 

CODE_HOME=~/fairseq/
DATA_PATH=~/data/glue-32768-fast
TENSORBOARD_LOG=/home/tianyuan/exps/fairseq/SE-F-SE-F-16-roberta_base-03768-p512-b256//GLUE/checkpoint_7_200000

cd $CODE_HOME

for SEED in $SEED_LIST
do
    for N_EPOCH in $N_EPOCH_LIST
    do
        for BATCH_SZ in $BATCH_SZ_LIST
        do
            SENT_PER_GPU=$(( BATCH_SZ / 1 ))
            N_UPDATES=$(( ((N_SENT + BATCH_SZ - 1) / BATCH_SZ) * N_EPOCH ))
            WARMUP_UPDATES=$(( N_UPDATES / 16 ))
            echo $SENT_PER_GPU $N_UPDATES $WARMUP_UPDATES
            for LR in $LR_LIST
            do
                for WEIGHT_DECAY in $WEIGHT_DECAY_LIST
                do

echo ${N_EPOCH} ${BATCH_SZ} ${LR} ${WEIGHT_DECAY} $SEED
OUTPUT_PATH=big-1213-${PROCEDURE_FOLDER}/GLUE/${CHECKPOINT_FILE}/${PROBLEM}/${N_EPOCH}-${BATCH_SZ}-${LR}-${WEIGHT_DECAY}-$SEED-0.006
mkdir -p $OUTPUT_PATH

if [ -e $OUTPUT_PATH/ok ]; then 
    continue
fi

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
FILE_NAME=$(basename $0)
cd $CODE_HOME
cp ${SHELL_FOLDER}/${FILE_NAME} ${OUTPUT_PATH}/${FILE_NAME}

python train.py $DATA_PATH/${PROBLEM}-bin \
       --restore-file $BERT_MODEL_PATH \
       --max-positions 512 \
       --max-sentences $SENT_PER_GPU \
       --max-tokens 4400 \
       --task sentence_prediction \
       --reset-optimizer --reset-dataloader --reset-meters \
       --required-batch-size-multiple 1 \
       --init-token 0 --separator-token 2 \
       --arch $ARCH \
       --criterion sentence_prediction $REGRESSION \
       --num-classes $N_CLASSES \
       --dropout 0.1 --attention-dropout 0.1 \
       --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
       --clip-norm 0.0 \
       --lr-scheduler polynomial_decay --lr $LR --total-num-update $N_UPDATES --warmup-updates $WARMUP_UPDATES\
       --max-epoch $N_EPOCH --seed $SEED --save-dir $OUTPUT_PATH --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints --no-best-checkpoints \
       --find-unused-parameters --skip-invalid-size-inputs-valid-test --truncate-sequence --embedding-normalize \
       --tensorboard-logdir $TENSORBOARD_LOG/${PROBLEM}/${N_EPOCH}-${BATCH_SZ}-${LR}-${WEIGHT_DECAY}-$SEED-0.006 --valid-subset valid,valid1\
       --best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric | tee -a $OUTPUT_PATH/train_log.txt

touch $OUTPUT_PATH/ok
rm -rf $OUTPUT_PATH/checkpoint_best.pt
rm -rf $OUTPUT_PATH/checkpoint_last.pt

                done
            done
        done
    done
done
