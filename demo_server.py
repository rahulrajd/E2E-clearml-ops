from fastapi import FastAPI,Request
import uvicorn
from preprocess_inference import Preprocess
from clearml import Task,Model
from config import *
app = FastAPI(debug=True)
import joblib
import shap
import json
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.get("/")
def home():
    return "server is Running"

@app.post("/check_payload/")
async def check_payload(payload:Request):
    data = await payload.json()
    response_df = pre.preprocess(body=data,state={})
    prediction = model.predict(response_df)
    shap_values = explainer.shap_values(response_df)
    return {"prediction":int(prediction[0])}

if __name__ == "__main__":
    task = Task.get_task(project_name=PROJECT_NAME,task_name="model_training")
    model = joblib.load(Model(model_id="bc5f742614aa4bb0baf68523244c5f2c").get_local_copy())
    explainer = shap.TreeExplainer(model)
    pre = Preprocess()
    uvicorn.run(app,port=5010)
