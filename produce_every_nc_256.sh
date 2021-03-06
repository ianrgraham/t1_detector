#!/bin/bash

#SBATCH --job-name=mass_prod
#SBATCH --output=/data1/shared/igraham/output/quasi_modes/%A-%a.out
#SBATCH --time=1-00:00:00
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-1000
#SBATCH --partition=compute
#SBATCH --mem=2G
#SBATCH --nodes=1

#20964

module load gcc/8.3.0
#module load hpcx/2.5.0/hpcx

#conda activate softmatter

ADD=20000

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    # test case
    NUM=1
else
    NUM=$(( ${SLURM_ARRAY_TASK_ID} + ${ADD} ))
fi

# execute c++ code for all files with N=2048
# mkdir -p /data1/shared/igraham/datasets/fast_sim2
cd /home1/igraham/Projects/t1_detector
root=/data1/shared/igraham/datasets/new_memory
dir=`sed "${NUM}q;d" 256_files.txt`
/home1/igraham/anaconda3/envs/softmatter/bin/python produce_voro_nc.py -d ${root}/${dir}
/home1/igraham/anaconda3/envs/softmatter/bin/python produce_t1s_nc.py -d ${root}/${dir}
/home1/igraham/anaconda3/envs/softmatter/bin/python produce_bd_nc.py -d ${root}/${dir}
