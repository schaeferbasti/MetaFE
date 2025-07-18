# import argparse
from src.Surrogate_Model import main


def run_metafe():
    # parser = argparse.ArgumentParser(description='Run CatBoost Surrogate Model with Metadata from Method')
    # parser.add_argument('--dataset', required=True, help='Metafeature Method')
    # args = parser.parse_args()
    dataset_id = 146818
    wanted_min_relative_improvement = 0.1
    memory_limit_mb = 64000
    time_limit_seconds = 600
    # main(int(args.dataset), wanted_min_relative_improvement, memory_limit_mb, time_limit_seconds)
    main(dataset_id, wanted_min_relative_improvement, memory_limit_mb, time_limit_seconds)


if __name__ == '__main__':
    run_metafe()
