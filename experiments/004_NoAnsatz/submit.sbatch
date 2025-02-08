#!/bin/bash

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=8
    #SBATCH --time=0:05:00
    #SBATCH --mem=32GB
    #SBATCH --gres=gpu:1
    #SBATCH --job-name=diffeo
    #SBATCH --mail-type=FAIL
    #SBATCH --mail-user=cm6627@nyu.edu
    #SBATCH --output=./slurm/slurm_%j.out
    #SBATCH --error=./slurm/slurm_%j.err
    
    module purge
    
    singularity exec --nv       --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:ro       --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro       --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro       --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro       /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif       /bin/bash -c "source /ext3/env.sh; python fit.py '{\"batch_size\": 16, \"epochs\": 100, \"model\": {\"name\": \"MLP_v2\", \"dropout_rate\": 0.02}, \"optimizer\": {\"name\": \"AdamW\", \"lr\": 0.0001, \"weight_decay\": 0.002}, \"scheduler\": {\"name\": \"ReduceLROnPlateau\", \"factor\": 0.1}, \"dataset\": \"resnet18_imagenet1k_train\"}'" 