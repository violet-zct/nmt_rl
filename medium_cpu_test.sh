#!/bin/bash

#echo $LD_LIBRARY_PATH
python -u enc_dec.py --dynet-mem 1500 --test --model_name "train.medium.concat_to_softmax" --concat_readout --hid_dim 256 --att_dim 256 --emb_size 256
