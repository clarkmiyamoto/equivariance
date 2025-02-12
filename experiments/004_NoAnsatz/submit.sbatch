#!/bin/bash

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=8
    #SBATCH --time=0:59:00
    #SBATCH --mem=32GB
    #SBATCH --gres=gpu:1
    #SBATCH --job-name=diffeo
    #SBATCH --mail-type=FAIL
    #SBATCH --mail-user=cm6627@nyu.edu
    #SBATCH --output=./slurm/slurm_%j.out
    #SBATCH --error=./slurm/slurm_%j.err
    
    module purge
    
    singularity exec --nv       --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:ro       --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro       --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro       --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro       /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif       /bin/bash -c "source /ext3/env.sh; python fit.py '{\"batch_size\": 64, \"epochs\": 1500, \"model\": {\"name\": \"MLP_v1\"}, \"optimizer\": {\"name\": \"Adam\", \"lr\": 1e-05}, \"scheduler\": {\"name\": \"None\"}, \"dataset\": \"resnet18_layer13_imagenet1ktrain_goldfishonly\"}'" 