set -ex

export MLU_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

horovodrun -np 8 -H localhost:8 python mlu_horovod_train.py 2>&1 | tee mlu_horovod_train_log
