import pandas as pd
import warnings
import json
import mlflow 
import optuna
import joblib
import tempfile

from pathlib import Path
from config import config
from dqAI import utils,train,predict
from argparse import Namespace
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback


warnings.filterwarnings('ignore')

def elt_data():
    """Extract, load and transform our data assets."""
    # Extract + Load
    df = pd.read_csv(config.DUPLICATE_QUESTIONS_URL,nrows=10000,error_bad_lines=False,sep=',')
    df = df.drop(['Unnamed: 0.1','Unnamed: 0','qid1','qid2'],axis=1)
    # df.to_csv(Path(config.DATA_DIR,'extracted_data.csv') ,index=False)
    # transform
    "any type of transfromation we can do here"
    print('data extraction complete')


def train_model(args_fp,experiment_name,run_name):
    """Train a model given arguments"""
    #load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "extracted_data.csv"),nrows=10000,error_bad_lines=False,sep=',')
    df = df.drop(['Unnamed: 0.1','Unnamed: 0','qid1','qid2'],axis=1)
    #train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)
    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))
       
def optimize(args_fp,study_name,num_trials):
    """Optimize hyperparameters."""
    args = Namespace(**utils.load_dict(filepath=args_fp))
    #load the data
    df = pd.read_csv(Path(config.DATA_DIR, "extracted_data.csv"),nrows=10000,error_bad_lines=False,sep=',')
    df = df.drop(['Unnamed: 0.1','Unnamed: 0','qid1','qid2'],axis=1)
    #optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback])
    #best trail
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")

def load_artifacts(run_id,model="sgd"):
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    data_dir = Path(config.DATA_DIR)
    # Load objects from run
    # args_fp = Path(config.CONFIG_DIR,"args.json")
    # args = Namespace(**utils.load_dict(filepath=args_fp))
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    if model == "sgd":
        model = joblib.load(Path(artifacts_dir, "model.pkl"))
    if model == 'randomforest':
        model = joblib.load(Path(data_dir, "randomforest.pkl"))
    performance = utils.load_dict(filepath=Path(config.CONFIG_DIR, "performance.json"))

    return {
        "args": args,
        "model": model,
        "performance": performance
    }

def predict_tag(q1,q2, run_id=None):
    """Predict tag for text."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id,model='randomforest')
    prediction = predict.predict(q1,q2, artifacts=artifacts)
    return prediction


