#!/bin/bash
MY_PYTHON=python
pyscript=main.py
exp_name="demo_MNIST"

# Paths
datapath="./data"
ofile="split_mnist.pt"

# Main Split-MNIST config
results="results/split_mnist_full/"
ds_args="--n_layers 2 --n_hiddens 400 --tasks_to_preserve 5 --save_path $results --data_path $datapath --log_every 500 --samples_per_task -1 --data_file $ofile --cuda yes"

# Appendix Split-MNIST-mini config
#results="results/split_mnist_mini/"
#ds_args="--n_layers 2 --n_hiddens 100 --tasks_to_preserve 5 --save_path $results --data_path $datapath --log_every 100 --samples_per_task 1000 --data_file $ofile --cuda yes"
mkdir 'results'
mkdir $results

cd "$datapath" || exit
cd raw/ || exit
$MY_PYTHON raw.py 'mnist' # Download
cd ..

# Prepare dataset
if [ ! -f $ofile ]; then
  echo "Preprocessing $ofile"
  $MY_PYTHON "mnist_split.py" \
    --o $ofile \
    --i "raw/" \
    --seed 0 \
    --n_tasks 5
fi
cd ..
##########################################################
# BALANCED method configs
##########################################################
# Methods: 'prototypical.CoPE', 'CoPE_CE', 'finetune', 'reservoir',  'gem', 'icarl', 'GSSgreedy'

# Grid over (pick best)
# n_outputs=128,256
# lr=0.05,0.01,0.005,0.001
n_memories=200 # Change for ablation (mem per class): 10,20,50,100,150,200

# CoPE
# Last code version acc check for 5 seeds: 94.110+-0.764 avg acc
# Note: Use the exact dependencies in README.md to reproduce the results.
# model="prototypical.CoPE"
# args="--model $model --batch_size 10 --lr 0.01 --loss_T 0.1 --p_momentum 0.99 --n_memories $n_memories --n_outputs 100 --n_iter 1 --n_seeds 5 $exp_name"
# $MY_PYTHON "$pyscript" $ds_args $args # Run python file

n_memories=200
model="DVC"
args="--model $model --batch_size 10 --lr 0.01 --n_memories $n_memories --n_outputs 10 --n_iter 1 --n_seeds 5 $exp_name --dl_weight 2.0 --subsample 50 --eps_mem_batch 10"
CUDA_VISIBLE_DEVICES=0 $MY_PYTHON "$pyscript" $ds_args $args # Run python file

# Cope-CE
model="CoPE_CE"
args="--model $model --batch_size 10 --lr 0.05 --n_memories $n_memories --n_outputs 10 --n_iter 1 --n_seeds 5 $exp_name"

# iid-offline
model="finetune"
args="--n_iter 1 --n_epochs 50 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid yes $exp_name"

# iid-online
model="finetune"
args="--n_iter 1 --n_epochs 1 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid yes $exp_name"

# finetune
model="finetune"
args="--n_iter 1 --n_epochs 1 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid no $exp_name"

# Reservoir
model="reservoir"
n_mem_tot=2000
args="--n_iter 1 --model $model --n_memories $n_mem_tot --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"

# iCaRL
n_mem_tot=2000
model="icarl"
args="--n_iter 1 --model $model --memory_strength 1 --n_memories $n_mem_tot --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 ICARL"

# GEM
model="gem"
n_mem_task=400 # 5 tasks
args="--n_iter 1 --model $model --memory_strength 0.5 --n_memories $n_mem_task --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"

# GSS
model="GSSgreedy"
n_mem_tot=2000
args="--model $model --batch_size 10 --lr 0.05 --n_memories 10 --n_sampled_memories $n_mem_tot --n_constraints 10 --memory_strength 10 --n_iter 1 --change_th 0. --subselect 1 --normalize no $exp_name"

# MIR
# See original implementation @ https://github.com/optimass/Maximally_Interfered_Retrieval
# Adapted to match settings in this paper

exit