import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def get_metrics(y_test,y_pred,classes=[0,1],capture_class=False):
    """Performance metrics using ground truths and predictions."""
    metrics = {"overall": {}, "class": {}}
    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_test, y_pred, average="binary")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    # metrics["overall"]["num_samples"] = np.float64(len(y_test))
    metrics['class'] = ["duplicate" for i in y_pred if i==1]
    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_test, y_pred, average=None)
    if capture_class:
        for i, _class in enumerate(classes):
            metrics['class'][_class] = {
                "precision": class_metrics[0][i],
                "recall": class_metrics[1][i],
                "f1": class_metrics[2][i],
                "num_samples": np.float64(class_metrics[3][i])
            }
    return metrics