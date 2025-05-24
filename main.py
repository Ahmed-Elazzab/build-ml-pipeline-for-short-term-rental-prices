import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]

@hydra.main(config_name='config', version_base='1.3')

def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
    
    root_path = hydra.utils.get_original_cwd()
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:
        #download step
        if "download" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
        #basic cleaning step
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Data with outliers and null values removed",
                "min_price": config['etl']['min_price'],
                "max_price": config['etl']['max_price']
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv.csv:latest",
                    "ref": "clean_sample.csv.csv:reference",
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        # split data for training and testing 
        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input": "nyc_airbnb/cleaned_data.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by']
                },
            )
                

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for 
            # the train_random_forest step                
            _ = mlflow.run(
                os.path.join(root_path, "src", "train_random_forest"),
                    "main",
                    parameters={
                        "trainval_artifact": "nyc_airbnb/trainval_data.csv:latest",
                        "val_size": config['modeling']['val_size'],
                        "random_seed": config['modeling']['random_seed'],
                        "stratify_by": config['modeling']['stratify_by'],
                        "rf_config": rf_config,
                        "max_tfidf_features": config['modeling']['max_tfidf_features'],
                        "output_artifact": config['modeling']['output_artifact']
                    },
            )            

        if "test_regression_model" in active_steps:
             _ = mlflow.run(
                    f"{config['main']['components_repository']}/test_regression_model",
                    "main",
                    parameters={
                        "mlflow_model": "nyc_airbnb/" + config['modeling']['output_artifact'] + ":prod",
                        "test_dataset": "nyc_airbnb/test_data.csv:latest"
                    },
            ) 


if __name__ == "__main__":
    go()