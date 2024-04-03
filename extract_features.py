import os

from qualitylib.tools import import_python_file
from videolib.standards import get_standard

from disque import DisqueFeatureExtractor

import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Code to extract features from video pair')
    parser.add_argument('--ref_video', help='Path to reference video', type=str)
    parser.add_argument('--dis_video', help='Path to distorted video', type=str)
    parser.add_argument('--fex_args', help='Path to Python file containing arguments to be passed to the feature extractor. Use fex_args to specify the model checkpoint.', type=str, default=None)
    parser.add_argument('--ref_standard', help='Standard to which the reference video conforms.', type=str)
    parser.add_argument('--dis_standard', help='Standard to which the distorted video conforms.', type=str)
    parser.add_argument('--width', help='Width of input video. Required for raw YUV videos. (Optional)', type=int, default=None)
    parser.add_argument('--height', help='Width of input video. Required for raw YUV videos. (Optional)', type=int, default=None)
    parser.add_argument('--framerate', help='Framerate of input video in FPS. Required for raw YUV videos. (Optional)', type=int, default=None)
    parser.add_argument('--out_file', help='Path to output MAT file containing results.', type=str)
    return parser


def main():
    args = get_parser().parse_args()
    asset_dict = {}
    asset_dict['dataset_name'] = None
    asset_dict['ref_path'] = args.ref_video
    asset_dict['dis_path'] = args.dis_video
    asset_dict['ref_standard'] = get_standard(args.ref_standard)
    asset_dict['dis_standard'] = get_standard(args.dis_standard)
    asset_dict['content_id'] = 0
    asset_dict['asset_id'] = 0
    asset_dict['score'] = None
    asset_dict['width'] = args.width
    asset_dict['height'] = args.height
    asset_dict['fps'] = args.framerate

    fex_args = []
    fex_kwargs = {}
    if args.fex_args is not None:
        mod = import_python_file(args.fex_args)
        if hasattr(mod, 'args'):
            fex_args.extend(mod.args)
        if hasattr(mod, 'kwargs'):
            fex_kwargs.update(mod.kwargs)

    fex = DisqueFeatureExtractor(*fex_args, use_cache=False, **fex_kwargs)
    result = fex(asset_dict)

    ext = os.path.splitext(args.out_file)[-1]
    if ext != 'mat':
        raise OSError(f'Invalid extension {ext}, expected \'mat\'')
    result.save(args.out_file)


if __name__ == '__main__':
    main()