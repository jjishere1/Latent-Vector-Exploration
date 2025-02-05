#!/bin/bash

nohup python main.py \
    --num_epochs 2 \
    --block_dims 5,5,5 \
    --padded_block_dims 9,9,9 \
    --use_padding \
    --output_vti output_500epochs_padding.vti \
    --loss_file training_loss_padding.txt \
    > my_run.out 2> my_run.err &

echo "Training started with PID: $!"
echo "Standard output is redirected to my_run.out"
echo "Standard error is redirected to my_run.err"
