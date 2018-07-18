#!/bin/bash

source /home/ts678/environments/python3/bin/activate


for FOLDER in mh4-local-s2s1s0b0 mh4t cle mh4t-two ahdp ah-local oneec;
do
    METHOD=`echo ${FOLDER} | cut -f 1 -d "-"`
    echo $METHOD

    for folder in /home/ts678/nonproj/models/cdparser/${FOLDER}/*;
    do
        echo $folder
        BASENAME=`basename "$folder"`
        LAN=`basename "$folder" | cut -f 1 -d "-"`
        INPUT=/home/ts678/nonproj/data/ud-treebanks-conll2017/${LAN}-ud-dev.conllu
        OUTPUT=/home/ts678/nonproj/data/output-dev/${BASENAME}-${FOLDER}.conllu
        MODEL=${folder}/model_0_model

        MKL_NUM_THREADS=1 python acl18-dev.py $INPUT $OUTPUT $MODEL $METHOD $LAN

    done
done
