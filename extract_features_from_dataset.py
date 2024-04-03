import argparse

from qualitylib.tools import import_python_file, read_dataset
from qualitylib.runner import Runner

from disque import DisqueFeatureExtractor


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run feature extractors and store results')
    parser.add_argument('--dataset', help='Path to dataset file for which to extract features', type=str)
    parser.add_argument('--fex_args', help='Path to Python file containing arguments to be passed to the feature extractor. Use fex_args to specify the model checkpoint.', type=str, default=None)
    parser.add_argument('--processes', help='Number of parallel processes', type=int, default=1)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    dataset = import_python_file(args.dataset)
    assets = read_dataset(dataset, shuffle=True)

    fex_args = []
    fex_kwargs = {}
    if args.fex_args is not None:
        mod = import_python_file(args.fex_args)
        if hasattr(mod, 'args'):
            fex_args.extend(mod.args)
        if hasattr(mod, 'kwargs'):
            fex_kwargs.update(mod.kwargs)

    runner = Runner(DisqueFeatureExtractor, *fex_args, processes=args.processes, use_cache=True, **fex_kwargs)  # Reads from stored results if available, else stores results.
    runner(assets, return_results=False)  # Only extract and save features, do not use for anything.


if __name__ == '__main__':
    main()