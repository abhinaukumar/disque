#!/bin/bash
for ind in $(seq 1 4)
do
    for sdr_ind in $(seq 1 4)
    do
        python3 process_image_using_example.py --ckpt_path DisQUE_Checkpoints/DisQUE_HDR.ckpt --source_range 1023 --target_range 255 --example_source_path DisQUE_Images/example_hdr_${ind}.png --example_target_path DisQUE_Images/example_sdr${sdr_ind}_${ind}.png --input_source_path DisQUE_Images/input_hdr_${ind}.png --output_target_path DisQUE_Images/output_sdr${sdr_ind}_${ind}.png
    done
done