from fastapi import FastAPI,Request
from http import HTTPStatus
from typing import Dict
from datetime import datetime
from functools import wraps
from pathlib import Path
from config import config
from dqAI import main , predict
from app.schemas import Text

app = FastAPI(
    title="DqAi",
    description="Classify quplicate questions",
    version="0.1"
)


@app.on_event("startup")
def load_artifacts():
    """
    we use the (@app.on_event("startup")) event to load the artifacts for the model to use for inference. 
    The advantage of doing this as an event is that our service won't start until this is complete and 
    so no requests will be prematurely processed and cause errors.
    """
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id,model='randomforest')
    print("Ready for inference!")

def construct_response(f):
    """Construct a JSON response for an endpoint."""
    """Adding meta data to responses"""
    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap

@app.get("/",tags=['General'])
@construct_response  ##decorater to add meta data to responses
def _index(request:Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response

@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = artifacts["performance"]
    data = {"performance":performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response

@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response

@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: Text) -> Dict:
    """Predict tags for a list of texts."""
    q1 = payload.q1
    q2 = payload.q2
    predictions = predict.predict(q1,q2,artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response