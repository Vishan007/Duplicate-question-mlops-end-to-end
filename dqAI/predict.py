import numpy as np
from dqAI import feature_extract

# Custom predict function
def custom_predict(y_prob, threshold, index):
    """Custom predict function that defaults
    to an index if conditions are not met."""
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(q1,q2, artifacts):
    """Predict tags for given texts."""
    x = feature_extract.query_point_creator(q1,q2)
    x = np.hstack((x,np.array([0]).reshape(1,-1)))
    y_pred_prob = artifacts["model"].predict_proba(x)
    y_pred = artifacts["model"].predict(x)
    if y_pred == 0:
        tex = 'not duplicate'
    else:
        tex = 'duplicate'
    predictions = [
        {
            "question1": q1,
            "question2": q2,
        },
        {
            "predicted_tag": f"{tex}"
        }
    ]
    return predictions