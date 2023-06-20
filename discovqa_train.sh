#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:a100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=12G
#SBATCH --time=0-04:00

source ~/projects/def-sshirmoh/mhashemi/exec/set_env.sh
cd ~/workspace/discovqa/

source <(cat .env | sed 's/^/export /')
export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

tensorboard --logdir $PROJECT_DIR/discovqa --host 0.0.0.0 --load_fast false &

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!
stmp01=`date +%s`

srun python trainer.py \\
--num_nodes $SLURM_NNODES \\
--num_workers $SLURM_CPUS_PER_TASK \\
--config config.toml \\
--prefix discovqa

stmp02=`date +%s`
echo "Execution Time: "$(($stmp02-$stmp01))