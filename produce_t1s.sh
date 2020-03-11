#!/bin/bash

#SBATCH --job-name=t1_prod
#SBATCH --output=/data1/shared/igraham/output/quasi_modes/%A-%a.out
#SBATCH --time=1-00:00:00
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=151-245
#SBATCH --partition=compute
#SBATCH --mem=2G
#SBATCH --nodes=1

module load gcc/8.3.0
#module load hpcx/2.5.0/hpcx

#conda activate softmatter

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    # test case
    NUM=1
else
    NUM=${SLURM_ARRAY_TASK_ID}
fi

# execute c++ code for all files with N=2048
# mkdir -p /data1/shared/igraham/datasets/fast_sim2
cd /home1/igraham/Projects/t1_detector
root=/home1/igraham/Projects/hoomd_test
dir=`sed "${NUM}q;d" ../quasilocalized_modes/process_files.txt`
/home1/igraham/anaconda3/envs/softmatter/bin/python produce_t1s.py -d ${root}/${dir}
