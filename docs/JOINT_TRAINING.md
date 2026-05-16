# Joint Multi-View Training

## Overview

The codebase now supports three modes for multi-view training in Phase 2:

### 1. **Alternative Training** (Default)
- Views are cycled sequentially every 100 steps
- Step 0-99: Train on view 1
- Step 100-199: Train on view 2
- Step 200-299: Train on view 3
- Step 300-399: Train on view 4
- Step 400+: Cycle repeats

### 2. **Joint Training** (Use `--joint_training` flag)
- All 4 views are processed together in the same batch
- Each training step sees all views simultaneously
- Effective batch size becomes `batch_size * num_frames`
- More memory-intensive but potentially better multi-view consistency

### 3. **Joint Training with Gradient Accumulation** (NEW - Recommended)
- Use `--joint_training --gradient_accumulation_steps 4`
- Processes views one at a time while accumulating gradients
- Mathematically equivalent to joint training
- **Memory usage: Same as single-view training**
- Best of both worlds: joint training benefits with minimal memory cost

## Usage

### Enable Joint Training with Gradient Accumulation (Recommended)
```bash
python dreamedit3d.py \
    --instance_data_dir="./data/multiview_data" \
    --joint_training \
    --gradient_accumulation_steps=4 \
    --num_frames=4 \
    --train_batch_size=1 \
    --resolution=256 \
    --mixed_precision="fp16" \
    ...
```

### Enable Joint Training (Original - High Memory)
```bash
python dreamedit3d.py \
    --instance_data_dir="./data/multiview_data" \
    --joint_training \
    --num_frames=4 \
    --train_batch_size=1 \
    ...
```

### Keep Alternative Training (Default)
```bash
python dreamedit3d.py \
    --instance_data_dir="./data/multiview_data" \
    --num_frames=4 \
    --train_batch_size=1 \
    ...
```

## Data Structure

Both modes expect multi-view data organized as:
```
instance_data_dir/
├── view_1/
│   ├── img.jpg
│   └── mask0.png
├── view_2/
│   ├── img.jpg
│   └── mask0.png
├── view_3/
│   ├── img.jpg
│   └── mask0.png
└── view_4/
    ├── img.jpg
    └── mask0.png
```

## Training Behavior

### Phase 1 (Textual Inversion)
- **Both modes**: Single view (view_1) only
- Learns token embeddings for the object
- Duration: `--phase1_train_steps` (default: 400)

### Phase 2 (Fine-tuning)
- **Alternative mode**: Cycles through views every 100 steps
- **Joint mode**: All views in every batch
- Fine-tunes UNet + token embeddings
- Duration: `--phase2_train_steps` (default: 400)

## Memory Considerations

### Joint Training with Gradient Accumulation (Recommended)
- **Memory usage**: Same as single-view training ✅
- **Training speed**: Slightly slower than full joint (processes views sequentially)
- **Quality**: Equivalent to full joint training
- **Recommended**: Use with high resolution (256+)
- **Example**: `--train_batch_size=1 --gradient_accumulation_steps=4 --resolution=256`

### Joint Training (Original)
- **Memory usage**: ~4x higher (processes all 4 views simultaneously)
- **Recommended**: Use smaller batch size or lower resolution
- **Example**: `--train_batch_size=1 --resolution=128`
- **Note**: May run out of memory on 24GB GPU at resolution 256

### Alternative Training
- **Memory usage**: Same as single-view training
- **Recommended**: Standard batch sizes work fine
- **Example**: `--train_batch_size=4`
- **Note**: Views trained separately, may have less multi-view consistency

## Expected Benefits

### Joint Training
✓ Better multi-view consistency (all views trained together)
✓ Faster convergence (sees all views every step)
✓ Potentially better 3D understanding

### Alternative Training
✓ Lower memory requirements
✓ More stable training
✓ Can use larger batch sizes

## Example Commands

### Joint Training with Gradient Accumulation (Recommended)
```bash
python dreamedit3d.py \
    --instance_data_dir="./data/chair" \
    --output_dir="./outputs/chair_joint_gradaccum" \
    --placeholder_token="<chair>" \
    --num_of_assets=1 \
    --num_frames=4 \
    --phase1_train_steps=400 \
    --phase2_train_steps=400 \
    --train_batch_size=1 \
    --learning_rate=2e-6 \
    --initial_learning_rate=5e-4 \
    --joint_training \
    --gradient_accumulation_steps=4 \
    --resolution=256 \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --mvdream_training_mode="2d" \
    --no_prior_preservation
```

### Joint Training (Original - High Memory)
```bash
python dreamedit3d.py \
    --instance_data_dir="./data/chair" \
    --output_dir="./outputs/chair_joint" \
    --placeholder_token="<chair>" \
    --num_of_assets=1 \
    --num_frames=4 \
    --phase1_train_steps=400 \
    --phase2_train_steps=400 \
    --train_batch_size=1 \
    --joint_training \
    --resolution=128 \
    --mixed_precision="fp16"
```

### Alternative Training (cycling views)
```bash
python dreamedit3d.py \
    --instance_data_dir="./data/chair" \
    --output_dir="./outputs/chair_alt" \
    --placeholder_token="<chair>" \
    --num_of_assets=1 \
    --num_frames=4 \
    --phase1_train_steps=400 \
    --phase2_train_steps=400 \
    --train_batch_size=4
```

## Technical Details

### How Joint Training with Gradient Accumulation Works (NEW)

1. **Data Loading**: Dataset loads all views at initialization
2. **Phase 1**: Uses only first view for textual inversion
3. **Phase 2 Transition**: `use_multiview=True` is set at step `phase1_train_steps`
4. **Gradient Accumulation Loop** (when `gradient_accumulation_steps > 1`):
   - For each view (1 to 4):
     - VAE encode: Convert image to latent space
     - Add noise: Apply forward diffusion
     - UNet forward: Predict noise
     - Compute loss: MSE between prediction and target
     - **Backward pass**: Accumulate gradients (no optimizer step yet)
     - Clear intermediate tensors to save memory
   - After all views: Single optimizer step with accumulated gradients
5. **Result**: Mathematically equivalent to processing all views together, but uses 1/4 the memory

### How Standard Joint Training Works

1. **Data Loading**: Dataset loads all views at initialization
2. **Phase 1**: Uses only first view for textual inversion
3. **Phase 2 Transition**: `use_multiview=True` is set at step `phase1_train_steps`
4. **Batch Processing**:
   - Images shape: `[batch_size, num_frames, C, H, W]`
   - Reshaped to: `[batch_size * num_frames, C, H, W]`
   - Text embeddings repeated for each view
5. **Loss Computation**: MSE loss computed across all views jointly

### Implementation Notes

- Joint training uses standard 2D UNet (not MVDream 3D attention)
- Each view is treated as an independent image with the same text conditioning
- Masks are properly handled for multi-view scenarios
- Attention loss is disabled in joint training mode (designed for single images)
- Gradient accumulation maintains numerical equivalence to full joint training

## Troubleshooting

### Out of Memory Error

**Solution 1: Use Gradient Accumulation (Recommended)**
```bash
--joint_training --gradient_accumulation_steps=4
```
This solves memory issues while maintaining joint training quality.

**Solution 2: Reduce Resolution**
- Lower `--resolution` to 128 or 64
- Use `--size` parameter to match resolution

**Solution 3: Other Memory Optimizations**
- Reduce `--train_batch_size` to 1
- Use `--mixed_precision="fp16"`
- Use `--use_8bit_adam`

### Dataloader Hangs
- Set `--dataloader_num_workers=0` (use main process)

### Poor Multi-View Consistency
- Try increasing `--phase2_train_steps`
- Experiment with `--learning_rate` (try 1e-6 to 5e-6)
- Use gradient accumulation for better multi-view training
