import argparse
import logging
import wandb
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    logger.info('downloading input artifact')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    sample = pd.read_csv(artifact_local_path)

    # drop outliers
    logger.info('drop outliers between price thresholds')
    idx = sample['price'].between(args.min_price, args.max_price)
    clean_sample = sample[idx].copy()

    logger.info('Save cleaned dataframe')
    clean_sample.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This step cleans the data')

    parser.add_argument('--input_artifact', type=str, required=True)
    parser.add_argument('--output_artifact', type=str, required=True)
    parser.add_argument('--output_type', type=str, required=True)
    parser.add_argument('--output_description', type=str, required=True)
    parser.add_argument('--min_price', type=float, required=True)
    parser.add_argument('--max_price', type=float, required=True)

    args = parser.parse_args()
    go(args)