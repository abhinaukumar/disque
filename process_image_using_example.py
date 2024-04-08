import argparse
import os

import cv2
import torch
from torchvision import transforms
import numpy as np

from disque import DisQUEModule
from disque.utils.png import read_png

def get_parser():
    parser = argparse.ArgumentParser(description='Example Guided Image Processing using DisQUE')
    parser.add_argument('--ckpt_path', help='Path to model checkpoint', type=str)
    parser.add_argument('--source_range', help='Range of pixel values in source domain (255 for 8-bit and 1023 for 10-bit)', type=int)
    parser.add_argument('--target_range', help='Range of pixel values in target domain (255 for 8-bit and 1023 for 10-bit)', type=int)
    parser.add_argument('--example_source_path', help='Path to example source image', type=str)
    parser.add_argument('--example_target_path', help='Path to example target image', type=str)
    parser.add_argument('--input_source_path', help='Path to input source image', type=str)
    parser.add_argument('--output_target_path', help='Path to output target image', type=str)
    return parser

def read_image(path):
    _, ext = os.path.splitext(path)
    if ext.lower() == '.png':
        img = read_png(path)  # Special handling because HDR images are provided as 16-bit PNG files.
    else:
        img = cv2.imread(path)[..., ::-1]
    return img

def main():
    args = get_parser().parse_args()
    module = DisQUEModule.load_from_checkpoint(args.ckpt_path, match_sizes=True, strict=True).cuda()
    to_tensor = transforms.ToTensor()

    example_source = read_image(args.example_source_path).astype('float64')
    example_target = read_image(args.example_target_path).astype('float64')
    input_source = read_image(args.input_source_path).astype('float64')

    height, width = example_source.shape[:2]
    pad_height = int(np.ceil(height / 16))*16
    pad_width = int(np.ceil(width / 16))*16

    x_ex = to_tensor(np.pad(example_source, [[0, pad_height-height], [0, pad_width-width], [0, 0]]) / args.source_range).float().cuda().unsqueeze(0)
    y_ex = to_tensor(np.pad(example_target, [[0, pad_height-height], [0, pad_width-width], [0, 0]]) / args.target_range).float().cuda().unsqueeze(0)
    x_in = to_tensor(np.pad(input_source, [[0, pad_height-height], [0, pad_width-width], [0, 0]]) / args.source_range).float().cuda().unsqueeze(0)

    with torch.no_grad():
        a_x_ex = module.appearance_enc(x_ex)
        a_y_ex = module.appearance_enc(y_ex)
        a_x_in = module.appearance_enc(x_in)
        c_x_in = module.content_enc(x_in)

        c_y_out = c_x_in
        a_y_out = tuple(a1 + a2 - a3 for a1, a2, a3 in zip(a_x_in, a_y_ex, a_x_ex))

        y_out = module.dec(c_y_out, a_y_out).squeeze().permute((1, 2, 0)).cpu().numpy()[:height, :width]

    output_target = np.clip(y_out, 0, 1)
    cv2.imwrite(args.output_target_path, (output_target[..., ::-1] * args.target_range))

if __name__ == '__main__':
    main()
