python train.py \
  --instance_data_dir /home/ai/gr/DreamEdit3D/examples/demo/thumbnails/4  \
  --num_of_assets 1 \
  --initializer_tokens head \
  --class_data_dir inputs/data_dir \
  --phase1_train_steps 400 \
  --phase2_train_steps 400 \
  --output_dir /home/ai/gr/DreamEdit3D/examples/demo/thumbnails/4 \
  --use_8bit_adam \
  --set_grads_to_none 