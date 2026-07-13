#!/bin/bash
# ==============================================================================
# submit_all_jobs.sh
# Automatically submits the entire pipeline to SLURM with correct dependencies.
# This guarantees scripts wait for each other to finish successfully.
# ==============================================================================

# Ensure logs directory exists
mkdir -p logs

echo "Submitting ARC SOTA Pipeline..."

# 1. Submit the setup script
SETUP_JOB_STR=$(sbatch arc_01_setup_sota.slurm)
# Example output: "Submitted batch job 12345"
SETUP_JOB_ID=$(echo $SETUP_JOB_STR | awk '{print $4}')
echo "✓ Setup Job: ${SETUP_JOB_ID}"

# 2. Submit SOTA prediction script (Depends on Setup)
PREDICT_JOB_STR=$(sbatch --dependency=afterok:${SETUP_JOB_ID} arc_02_predict_sota.slurm)
PREDICT_JOB_ID=$(echo $PREDICT_JOB_STR | awk '{print $4}')
echo "✓ SOTA Predict Job: ${PREDICT_JOB_ID} (Waiting on Setup)"

# 3. Submit your 5-seed Training Array (Independent of Setup, can run concurrently)
# --array=0-29 covers all 2 models × 3 ontologies × 5 seeds = 30 combinations
TRAIN_JOB_STR=$(sbatch --array=0-29 arc_submit_5seeds.slurm)
TRAIN_JOB_ID=$(echo $TRAIN_JOB_STR | awk '{print $4}')
echo "✓ 5-Seed Training Array: ${TRAIN_JOB_ID}"

# 4. Submit Evaluation & Plotting (Depends on BOTH SOTA Predict and 5-Seed Training)
EVAL_JOB_STR=$(sbatch --dependency=afterok:${PREDICT_JOB_ID}:${TRAIN_JOB_ID} arc_03_evaluate_and_plot.slurm)
EVAL_JOB_ID=$(echo $EVAL_JOB_STR | awk '{print $4}')
echo "✓ Evaluate & Plot Job: ${EVAL_JOB_ID} (Waiting on SOTA Predict & Training Array)"

echo ""
echo "All jobs submitted successfully! You can safely log off now."
echo "Use 'squeue -u \$USER' to monitor them."
