import pandas as pd
import numpy as np
import json
import mlflow
import optuna

from pathlib import Path
from tqdm import tqdm
from config import config
from dqAI import feature_extract, utils,evaluate,data
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, precision_recall_fscore_support

def train(args,df,trial=None,clean=False,create_feature=False):
    #setup
    utils.set_seeds()
    if clean:
        df.questio1 = df.question1.apply(data.preprocess)  ## data is already preprocessed
        df.questio1 = df.question1.apply(data.preprocess)  ## data is already preprocessed
    
    ##Feature extraction and tfidf Vectorization
    if create_feature:
        feature_arr = []
        for q1,q2 in tqdm(zip(df.question1.values,df.question2.values)):
            feature_arr.append(feature_extract.query_point_creator(q1,q2))
        feature_arr=np.array(feature_arr)
        feature_arr=feature_arr.reshape(10000,6022)
    X = pd.read_csv(Path(config.DATA_DIR, "feature.csv"))
    y = df.is_duplicate
    
    #train_test_split
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(X,y)
    
    #model
    model = SGDClassifier(
            loss="log", penalty="l2", alpha=args.alpha, max_iter=1,
            learning_rate="constant", eta0=args.learning_rate, power_t=args.power_t, warm_start=True)

#     model = RandomForestClassifier(max_features=args.max_features, min_samples_split=args.min_samples_split,
#                        n_estimators=args.n_estimators)
    
    #training
    for epoch in tqdm(range(args.num_epochs)):
        model.fit(X_train, y_train)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))
        if not epoch%10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )
        # ml-flow Log
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            
#         Pruning (for optimization )
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()   
    # Threshold
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    args.threshold = np.quantile([y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)  # Q1
    
    # Evaluation
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    # y_pred = utils.custom_predict(y_prob=y_prob, threshold=args.threshold, index=0)
    metrics = precision_recall_fscore_support(y_test, y_pred, average="binary")
    performance = evaluate.get_metrics(y_test=y_test,y_pred=y_pred)
    # print (json.dumps(performance, indent=2))
    
    return {
        "args": args,
        "model": model,
        "performance": performance
    } 

# tagifai/train.py
def objective(args, df, trial):
    """Objective function for optimization trials."""
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    print(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]