#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH -t 0

module load cuda-8.0 cudnn-8.0-5.1

export LD_LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH:$BOOST_LIBDIR
export CPATH=/opt/cudnn-8.0/include:$CPATH
export LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH:$BOOST_LIBDIR

#echo $LD_LIBRARY_PATH
python -u enc_dec.py --dynet-mem 10000 --dynet-gpu --model_name "train.mlp_to_softmax" 2>train.log
