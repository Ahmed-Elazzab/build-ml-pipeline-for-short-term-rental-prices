#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""

import argparse
import logging
import wandb
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    """
    1- Download input artifact. This will also log that this script is using this
    2- particular version of the artifact
    3- artifact_local_path = run.use_artifact(args.input_artifact).file()
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    logger.info(f'downloading input artifact')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    sample = pd.read_csv(artifact_local_path)


    logger.info('downloading input artifact')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    sample = pd.read_csv(artifact_local_path)
    
    # drop outliers
    logger.info(f'drop outliers between price thresholds')
    min_price = args.min_price
    max_price = args.max_price
    idx = sample['price'].between(min_price, max_price)
    clean_sample = sample[idx].copy()
    

    logger.info(f'Save cleaned dataframe')
    clean_sample.to_csv(args.output_artifact_name, index=False)

    artifact = wandb.Artifact(
    args.output_artifact,
    type=args.output_type,
    description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This step cleans the data')

    parser.add_argument(
        '--input_artifact', 
        type=str,
        help='Input artifact as given (sample csv file)',
        required=True
    )

    parser.add_argument(
        '--output_artifact_name', 
        type=str,
        help='Output file name (e.g. cleaned_data.csv)',
        required=True
    )

    parser.add_argument(
        '--output_artifact_type', 
        type=str,
        help='Type of the output file (e.g. cleaned_data)',
        required=True
    )

    parser.add_argument(
        '--output_artifact_description', 
        type=str,
        help='Cleaned data (e.g. outliers, skewness removed; datetime convertion)',
        required=True
    )

    parser.add_argument(
        '--min_price', 
        type=float,
        help='Minimum price to filter the price data for (e.g 10)',
        required=True
    )

    parser.add_argument(
        '--max_price', 
        type=float,
        help='Maximum price to filter the price data for (e.g 350)',
        required=True
    )
    
    args = parser.parse_args()

    go(args)