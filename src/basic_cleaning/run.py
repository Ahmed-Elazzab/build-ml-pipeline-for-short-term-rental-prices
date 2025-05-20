#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
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
        this function should do as follows
        1-read-in file
        2-Download input artifact. This will also log that this script is using this
        3-particular version of the artifact
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)


    input_artifact_name = f"raw_data:v{args.parameter1}" 
    logger.info(f"Downloading input artifact: {input_artifact_name}")
    artifact = run.use_artifact(input_artifact_name)
    artifact_path = artifact.file()

    logger.info("Reading input file")
    df = pd.read_csv(artifact_path)

    df.dropna(inplace=True)

    # Using parameter2 as part of the output filename
    output_path = f"cleaned_data_version_{args.parameter2}.csv"
    logger.info(f"Saving cleaned data")
    df.to_csv(output_path, index=False)

    logger.info("Logging output artifact")
    artifact = wandb.Artifact(
        name=f"cleaned_data_v{args.parameter2}", # Using parameter2 for the artifact name
        type="cleaned_data",
        description=args.parameter3,
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--parameter1", 
        type= int, # Changed to int
        help= "The version number of the input raw data artifact (e.g., 1 for 'raw_data:v1')",
        required=True
    )

    parser.add_argument(
        "--parameter2", 
        type= int, # Changed to int
        help= "A version number for the output cleaned data artifact",
        required=True
    )

    parser.add_argument(
        "--parameter3", 
        type= str,
        help= "A brief description of the cleaning process or output data",
        required=True
    )


    args = parser.parse_args()

    go(args)