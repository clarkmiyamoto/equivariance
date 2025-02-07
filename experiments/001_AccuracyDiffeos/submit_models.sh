#!/bin/bash

# List of model names
model_names=(
  'alexnet' 'convnext_base' 'convnext_large' 'convnext_small' 'convnext_tiny' 'densenet121' 
  'densenet161' 'densenet169' 'densenet201' 'efficientnet_b0' 'googlenet' 'maxvit_t' 
  'mnasnet0_5' 'mnasnet0_75' 'mnasnet1_0' 'mnasnet1_3' 'mobilenet_v2' 'mobilenet_v3_large' 
  'mobilenet_v3_small' 'regnet_x_16gf' 'regnet_x_1_6gf' 'regnet_x_32gf' 'regnet_x_3_2gf' 
  'regnet_x_400mf' 'regnet_x_800mf' 'regnet_x_8gf' 'regnet_y_16gf' 'regnet_y_1_6gf' 
  'regnet_y_32gf' 'regnet_y_3_2gf' 'regnet_y_400mf' 'regnet_y_800mf' 'regnet_y_8gf' 
  'resnet101' 'resnet152' 'resnet18' 'resnet34' 'resnet50' 'resnext101_32x8d' 
  'resnext101_64x4d' 'resnext50_32x4d' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0' 
  'shufflenet_v2_x1_5' 'shufflenet_v2_x2_0' 'squeezenet1_0' 'squeezenet1_1' 'swin_b' 
  'swin_s' 'swin_t' 'vgg11' 'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 
  'vgg19_bn' 'vit_b_16' 'vit_b_32' 'vit_l_16' 'vit_l_32' 'wide_resnet101_2' 'wide_resnet50_2'
)

# Base directory for generated scripts
script_dir="./sbatch_scripts"
mkdir -p $script_dir

# Loop over each model name
for model_name in "${model_names[@]}"; do
  script_file="${script_dir}/${model_name}.sbatch"

  # Create the .sbatch file
  cat <<EOT > $script_file
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=${model_name}
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

singularity exec --nv \
  --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; python simulation.py --model_name ${model_name} --num_images 2000"
EOT

  # Submit the .sbatch file
  sbatch $script_file
done

