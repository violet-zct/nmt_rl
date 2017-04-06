#!/bin/bash
python -u analysis.py --dynet-mem 10000 --model_name "train.medium.concat_to_softmax" --concat_readout --hid_dim 256 --att_dim 256 --emb_size 256 2>test_concat_medium.log
