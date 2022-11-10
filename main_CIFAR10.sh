#!/bin/bash
MY_PYTHON=python
pyscript=main.py
exp_name="demo_CIFAR10"

# Paths
datapath="./data"
ofile="cifar10.pt"

# Main Split-CIFAR10 config
results="results/cifar10/"
ds_args="--tasks_to_preserve 5 --save_path $results --data_path $datapath --log_every 100 --samples_per_task 10000 --data_file $ofile --cuda yes"

# Appendix Split-CIFAR10-mini config
#results="results/cifar10mini/"
#ds_args="--tasks_to_preserve 5 --save_path $results --data_path $datapath --log_every 100 --samples_per_task 2000 --data_file $ofile --cuda yes"
mkdir 'results'
mkdir $results

cd "$datapath" || exit
cd raw/ || exit
$MY_PYTHON raw.py 'cifar10' # Download
cd ..

# Prepare dataset
if [ ! -f $ofile ]; then
  echo "Preprocessing $ofile"
  $MY_PYTHON "cifar10.py" \
    --o $ofile \
    --i "raw/cifar10.pt" \
    --seed 0 \
    --n_tasks 5
fi
cd ..

# ##########################################################
# # BALANCED method configs
# ##########################################################
# # Methods: 'prototypical.CoPE', 'CoPE_CE', 'finetune', 'reservoir',  'gem', 'icarl', 'GSSgreedy'

# # Grid over (pick best)
# # n_outputs=128,256
# # lr=0.05,0.01,0.005,0.001
n_memories=100 # Change for ablation (mem per class): 10,20,50,100,150,200

# # CoPE
# # Last code version acc check for 5 seeds: 49.610+-3.441 avg acc
# # Avg accs per seed=[tensor([0.4576]), tensor([0.5106]), tensor([0.5040]), tensor([0.4664]), tensor([0.5419])]
# # Note: Use the exact dependencies in README.md to reproduce the results.

# n_memories=100 
# model="prototypical.CoPE"
# args="--model $model --batch_size 10 --lr 0.005 --loss_T 0.1 --p_momentum 0.99 --n_memories $n_memories --n_outputs 256 --n_iter 1 --n_seeds 5 $exp_name"
# CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file
for lr in 0.005 0.003 0.01 0.05 0.1 0.2; do 
  for dl_weight in 2.0 2.5 3.0 3.5; do
    n_memories=1000
    model="DVC"
    args="--model $model --batch_size 10 --lr $lr --loss_T 0.1 --n_memories $n_memories --n_outputs 10 --n_iter 1 --n_seeds 5 $exp_name --dl_weight $dl_weight --subsample 50 --eps_mem_batch 10"
    CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file

    n_memories=200
    model="DVC"
    args="--model $model --batch_size 10 --lr $lr --loss_T 0.1 --n_memories $n_memories --n_outputs 10 --n_iter 1 --n_seeds 5 $exp_name --dl_weight $dl_weight --subsample 50 --eps_mem_batch 10"
    CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file

    n_memories=500
    model="DVC"
    args="--model $model --batch_size 10 --lr $lr --loss_T 0.1 --n_memories $n_memories --n_outputs 10 --n_iter 1 --n_seeds 5 $exp_name --dl_weight $dl_weight --subsample 50 --eps_mem_batch 10"
    CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file
    
   done
 done 
 
 
for lr in 0.01 0.05 0.1; do
  for ER_weight in 0.3 0.5 0.8 1; do

    n_memories=500
    model="OCD"
    args="--model $model --batch_size 32 --lr $lr  --n_memories $n_memories --n_outputs 10 --n_iter 5 --n_seeds 5 $exp_name --minibatch_size 32 --Bernoulli_probability 0.2 --ER_weight $ER_weight"
    CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file

    n_memories=200
    model="OCD"
    args="--model $model --batch_size 32 --lr $lr  --n_memories $n_memories --n_outputs 10 --n_iter 5 --n_seeds 5 $exp_name --minibatch_size 32 --Bernoulli_probability 0.2 --ER_weight $ER_weight"
    CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file

    n_memories=1000
    model="OCD"
    args="--model $model --batch_size 32 --lr $lr  --n_memories $n_memories --n_outputs 10 --n_iter 5 --n_seeds 5 $exp_name --minibatch_size 32 --Bernoulli_probability 0.2 --ER_weight $ER_weight"
    CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file
    
  done
done


#_____________________________________________________________________

# # Cope-CE
# model="CoPE_CE"
# args="--model $model --batch_size 10 --lr 0.05 --n_memories $n_memories --n_outputs 10 --n_iter 1 --n_seeds 5 $exp_name"

# # iid-offline
# model="finetune"
# args="--n_iter 1 --n_epochs 50 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid yes $exp_name"

# # iid-online
# model="finetune"
# args="--n_iter 1 --n_epochs 1 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid yes $exp_name"

# # finetune
# model="finetune"
# args="--n_iter 1 --n_epochs 1 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid no $exp_name"

# Reservoir
# model="reservoir"
# n_mem_tot=500
# args="--n_iter 1 --model $model --n_memories $n_mem_tot --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"
# CUDA_VISIBLE_DEVICES=1 $MY_PYTHON "$pyscript" $ds_args $args # Run python file
# # iCaRL
# n_mem_tot=1000
# model="icarl"
# args="--n_iter 1 --model $model --memory_strength 1 --n_memories $n_mem_tot --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 ICARL"

# # GEM
# model="gem"
# n_mem_task=200 # 5 tasks
# args="--n_iter 1 --model $model --memory_strength 0.5 --n_memories $n_mem_task --lr 0.005 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"

# # GSS
# model="GSSgreedy"
# n_mem_tot=1000
# args="--model $model --batch_size 10 --lr 0.05 --n_memories 10 --n_sampled_memories $n_mem_tot --n_constraints 10 --memory_strength 10 --n_iter 1 --change_th 0. --subselect 1 --normalize no $exp_name"

# # MIR
# # See original implementation @ https://github.com/optimass/Maximally_Interfered_Retrieval
# # Adapted to match settings in this paper

# exit

