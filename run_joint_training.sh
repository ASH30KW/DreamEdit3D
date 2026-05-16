#!/bin/bash

# Example script for running joint multi-view training
# This will train on all 4 views together in each batch (instead of alternating between views)

python dreamedit3d.py \
    --instance_data_dir="./data/multiview_data" \
    --output_dir="./outputs/joint_training" \
    --placeholder_token="<asset>" \
    --num_of_assets=1 \
    --num_frames=4 \
    --phase1_train_steps=400 \
    --phase2_train_steps=400 \
    --train_batch_size=1 \
    --learning_rate=2e-6 \
    --initial_learning_rate=5e-4 \
    --joint_training \
    --mvdream_training_mode="2d" \
    --resolution=256 \
    --mixed_precision="fp16" \
    --gradient_checkpointing \
    --img_log_steps=100

# Key differences from alternative training:
# --joint_training: Enable joint multi-view training (all views processed together)
#
# With joint training:
# - Phase 1: Standard 2D textual inversion on single view
# - Phase 2: All 4 views are loaded and processed together in each batch
#   * Each batch will contain batch_size * num_frames images
#   * All views share the same text conditioning
#   * Loss is computed across all views simultaneously
#
# Without joint training (default alternative mode):
# - Phase 1: Standard 2D textual inversion
# - Phase 2: Views are cycled every 100 steps (view 1 → view 2 → view 3 → view 4 → repeat)
