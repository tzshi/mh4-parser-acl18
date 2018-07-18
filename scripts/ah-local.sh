#!/bin/bash

source /home/ts678/environments/python3/bin/activate

MAX_STEPS=100000
EVAL_STEPS=500
DECAY_EVALS=5
DECAY_TIMES=3

BATCH_SIZE=100
OPTIMIZER=adam
LEARNING_RATE=2e-3
BETA1=0.9
BETA2=0.999
EPSILON=1e-8
SPARSE_UPDATES=False

CUTOFF=5
WORD_SMOOTH=2.5

CLIP=5.0

WDIMS=128
CDIMS=64
PDIMS=0
FDIMS=0

FEATURE_DROPOUT=0.

BILSTM_DIMS=192
BILSTM_LAYERS=2
BILSTM_DROPOUT=0.3

BIDIRECTIONAL=True

CLSTM_DIMS=128
CLSTM_LAYERS=2
CLSTM_DROPOUT=0.3

CHAR_REPR_METHOD=pred

BLOCK_DROPOUT=0.1
CHAR_DROPOUT=0.

UTAGGER_MLP_ACTIVATION=tanh
UTAGGER_MLP_DIMS=192
UTAGGER_MLP_LAYERS=1
UTAGGER_MLP_DROPOUT=0.3
UTAGGER_DISCRIM=True

XTAGGER_MLP_ACTIVATION=tanh
XTAGGER_MLP_DIMS=192
XTAGGER_MLP_LAYERS=1
XTAGGER_MLP_DROPOUT=0.3
XTAGGER_DISCRIM=True

AH_MLP_ACTIVATION=tanh
AH_MLP_DIMS=192
AH_MLP_LAYERS=1
AH_MLP_DROPOUT=0.3
AH_GLOBAL=False

AE_MLP_ACTIVATION=tanh
AE_MLP_DIMS=192
AE_MLP_LAYERS=1
AE_MLP_DROPOUT=0.3
AE_GLOBAL=False

AS_MLP_ACTIVATION=tanh
AS_MLP_DIMS=192
AS_MLP_LAYERS=1
AS_MLP_DROPOUT=0.3

MST_MLP_ACTIVATION=tanh
MST_MLP_DIMS=192
MST_MLP_LAYERS=1
MST_MLP_DROPOUT=0.3
MST_DISCRIM=True
MST_TRAINCLE=False

NPMST_MLP_ACTIVATION=tanh
NPMST_MLP_DIMS=192
NPMST_MLP_LAYERS=1
NPMST_MLP_DROPOUT=0.3
NPMST_DISCRIM=True

MH4_MLP_ACTIVATION=tanh
MH4_MLP_DIMS=192
MH4_MLP_LAYERS=1
MH4_MLP_DROPOUT=0.3
MH4_DISCRIM=True
MH4_TRAINCLE=False

MH4T_MLP_ACTIVATION=tanh
MH4T_MLP_DIMS=192
MH4T_MLP_LAYERS=1
MH4T_MLP_DROPOUT=0.3
MH4T_MODE=local
MH4T_STACK_FEATURES=2
MH4T_BUFFER_FEATURES=1

LABEL_MLP_ACTIVATION=tanh
LABEL_MLP_DIMS=192
LABEL_MLP_LAYERS=1
LABEL_MLP_DROPOUT=0.3
LABEL_DISCRIM=True

AHDP_WEIGHT=1.0
AEDP_WEIGHT=1.0
AS_WEIGHT=1.0
MST_WEIGHT=1.0
NPMST_WEIGHT=1.0
MH4_WEIGHT=1.0
MH4T_WEIGHT=1.0
LABEL_WEIGHT=1.0
UTAGGER_WEIGHT=0.2
XTAGGER_WEIGHT=0.2

AHDP_NUM=1
AEDP_NUM=0
MST_NUM=0
NPMST_NUM=0
MH4_NUM=0
MH4T_NUM=0
UTAGGER_NUM=1
XTAGGER_NUM=1

LABEL_BIAFFINE=False

RUN=$2
LANGUAGE=$1

LOG_FOLDER=/home/ts678/nonproj/models/cdparser/ah-local/

mkdir -p $LOG_FOLDER

SAVE_PREFIX=${LOG_FOLDER}/${LANGUAGE}-$RUN

mkdir -p $SAVE_PREFIX

LOG_FILE=$SAVE_PREFIX/log.log

TRAIN_FILE=/home/ts678/nonproj/data/ud-treebanks-conll2017/${LANGUAGE}-ud-train.conllu
DEV_FILE=/home/ts678/nonproj/data/ud-treebanks-conll2017/${LANGUAGE}-ud-dev.conllu


MKL_NUM_THREADS=1 python -m acl18.cdparser \
    build-vocab $TRAIN_FILE --cutoff ${CUTOFF} \
    - create-parser --batch-size $BATCH_SIZE --word-smooth $WORD_SMOOTH \
        --clip $CLIP --block-dropout $BLOCK_DROPOUT --char-dropout $CHAR_DROPOUT \
        --learning-rate $LEARNING_RATE --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON \
        --optimizer $OPTIMIZER \
        --sparse-updates $SPARSE_UPDATES \
        --wdims $WDIMS --cdims $CDIMS --pdims $PDIMS \
        --fdims $FDIMS --feature-dropout $FEATURE_DROPOUT \
        --bidirectional $BIDIRECTIONAL \
        --bilstm-dims $BILSTM_DIMS --bilstm-layers $BILSTM_LAYERS --bilstm-dropout $BILSTM_DROPOUT \
        --char-lstm-dims $CLSTM_DIMS --char-lstm-layers $CLSTM_LAYERS --char-lstm-dropout $CLSTM_DROPOUT \
        --char-repr-method $CHAR_REPR_METHOD \
        --utagger-mlp-activation $UTAGGER_MLP_ACTIVATION --utagger-mlp-dims $UTAGGER_MLP_DIMS \
        --utagger-mlp-layers $UTAGGER_MLP_LAYERS --utagger-mlp-dropout $UTAGGER_MLP_DROPOUT \
        --utagger-discrim $UTAGGER_DISCRIM \
        --utagger-num $UTAGGER_NUM \
        --xtagger-mlp-activation $XTAGGER_MLP_ACTIVATION --xtagger-mlp-dims $XTAGGER_MLP_DIMS \
        --xtagger-mlp-layers $XTAGGER_MLP_LAYERS --xtagger-mlp-dropout $XTAGGER_MLP_DROPOUT \
        --xtagger-discrim $XTAGGER_DISCRIM \
        --xtagger-num $XTAGGER_NUM \
        --ah-mlp-activation $AH_MLP_ACTIVATION --ah-mlp-dims $AH_MLP_DIMS --ah-mlp-layers $AH_MLP_LAYERS \
        --ah-mlp-dropout $AH_MLP_DROPOUT \
        --ahdp-num $AHDP_NUM \
        --ah-global $AH_GLOBAL \
        --ae-mlp-activation $AE_MLP_ACTIVATION --ae-mlp-dims $AE_MLP_DIMS --ae-mlp-layers $AE_MLP_LAYERS \
        --ae-mlp-dropout $AE_MLP_DROPOUT \
        --aedp-num $AEDP_NUM \
        --ae-global $AE_GLOBAL \
        --as-mlp-activation $AS_MLP_ACTIVATION --as-mlp-dims $AS_MLP_DIMS --as-mlp-layers $AS_MLP_LAYERS \
        --as-mlp-dropout $AS_MLP_DROPOUT \
        --mst-mlp-activation $MST_MLP_ACTIVATION --mst-mlp-dims $MST_MLP_DIMS \
        --mst-mlp-layers $MST_MLP_LAYERS --mst-mlp-dropout $MST_MLP_DROPOUT \
        --mst-discrim $MST_DISCRIM \
        --mst-num $MST_NUM \
        --mst-traincle $MST_TRAINCLE \
        --npmst-mlp-activation $NPMST_MLP_ACTIVATION --npmst-mlp-dims $NPMST_MLP_DIMS \
        --npmst-mlp-layers $NPMST_MLP_LAYERS --npmst-mlp-dropout $NPMST_MLP_DROPOUT \
        --npmst-discrim $NPMST_DISCRIM \
        --npmst-num $NPMST_NUM \
        --mh4-mlp-activation $MH4_MLP_ACTIVATION --mh4-mlp-dims $MH4_MLP_DIMS \
        --mh4-mlp-layers $MH4_MLP_LAYERS --mh4-mlp-dropout $MH4_MLP_DROPOUT \
        --mh4-discrim $MH4_DISCRIM \
        --mh4-num $MH4_NUM \
        --mh4-traincle $MH4_TRAINCLE \
        --mh4t-mlp-activation $MH4T_MLP_ACTIVATION --mh4t-mlp-dims $MH4T_MLP_DIMS \
        --mh4t-mlp-layers $MH4T_MLP_LAYERS --mh4t-mlp-dropout $MH4T_MLP_DROPOUT \
        --mh4t-num $MH4T_NUM \
        --mh4t-mode $MH4T_MODE \
        --mh4t-stack-features $MH4T_STACK_FEATURES --mh4t-buffer-features $MH4T_BUFFER_FEATURES \
        --label-mlp-activation $LABEL_MLP_ACTIVATION --label-mlp-dims $LABEL_MLP_DIMS \
        --label-mlp-layers $LABEL_MLP_LAYERS --label-mlp-dropout $LABEL_MLP_DROPOUT \
        --label-discrim $LABEL_DISCRIM \
        --label-biaffine $LABEL_BIAFFINE \
        --mst-weight $MST_WEIGHT --ahdp-weight $AHDP_WEIGHT --aedp-weight $AEDP_WEIGHT \
        --npmst-weight $NPMST_WEIGHT \
        --mh4-weight $MH4_WEIGHT \
        --mh4t-weight $MH4T_WEIGHT \
        --as-weight $AS_WEIGHT \
        --label-weight $LABEL_WEIGHT --utagger_weight $UTAGGER_WEIGHT --xtagger-weight $XTAGGER_WEIGHT \
    - init-model \
    - train $TRAIN_FILE --dev $DEV_FILE \
        --utag True --xtag True --ast False --ahdp True --aedp False --mst False --npmst False --mh4 False --mh4t False --label True \
        --dev-portion 1.0 \
        --save-prefix $SAVE_PREFIX/model \
        --max-steps $MAX_STEPS --eval-steps $EVAL_STEPS --decay-evals $DECAY_EVALS --decay-times $DECAY_TIMES \
    - finish --dynet-mem 2000 --dynet-autobatch 0 \
&> $LOG_FILE
# --dynet-seed ${RUN}
