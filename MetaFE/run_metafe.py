# import argparse
from MetaFE.Surrogate_Model import main


def run_metafe():
    # parser = argparse.ArgumentParser(description='Run CatBoost Surrogate Model with Metadata from Method')
    # parser.add_argument('--dataset', required=True, help='Metafeature Method')
    # args = parser.parse_args()
    datasets = [2073, 146818, 146820, 167120, 167210, 168350, 168757, 168784, 189354, 190146, 233211, 359930, 359931, 359932, 359933, 359935, 359936, 359937, 359938, 359944, 359949, 359950, 359952, 359954, 359955, 359956, 359958, 359959, 359960, 359962, 359963, 359965, 359968, 359971, 359972, 359974, 359975, 359979, 359981, 359982, 359983, 359987, 359992, 359993]
    wanted_min_relative_improvement = 0.1
    memory_limit_mb = 64000
    time_limit_seconds = 1800
    for dataset_id in datasets:
        # main(int(args.dataset), wanted_min_relative_improvement, memory_limit_mb, time_limit_seconds)
        main(dataset_id, wanted_min_relative_improvement, memory_limit_mb, time_limit_seconds)


if __name__ == '__main__':
    run_metafe()
