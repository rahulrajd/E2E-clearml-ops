from typing import Any
import pandas as pd
import numpy as np
from clearml import Task,Model

import joblib
PROJECT_NAME = "Loan Approval V3"
# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        preprocessing_task = Task.get_task(project_name=PROJECT_NAME,task_name="preprocess_data")
        self.encoder = joblib.load(preprocessing_task.get_models()["output"][0].get_local_copy())


    def preprocess(self, body: dict,state: dict, collect_custom_statistics_fn=None) -> dict:
        #print(body)
        ordered_payload = {}
        _key = body.keys()
        for key in _key:
            if key not in ordered_payload:
                ordered_payload[key] = body[key]
        df = pd.DataFrame.from_dict(ordered_payload)
        num_columns = df[df.select_dtypes(exclude=["object"]).columns]
        category_columns = df[df.select_dtypes(include=["object"]).columns]
        cat_df = self.encoder.transform(category_columns)
        pred_data = pd.concat([cat_df,num_columns],axis=1)

        return pred_data


    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=data)


if __name__ == "__main__":
    payload = {"Gender": ["Male"], "Married": ["Yes"], "Dependents": [0], "Education": ["Graduate"], "Self_Employed": ["Yes"], "ApplicantIncome": [0], "CoapplicantIncome": [0], "LoanAmount": [0], "Loan_Amount_Term": [24], "Credit_History": [1], "Property_Area": ["Urban"]}
    c = Preprocess()
    c.preprocess(body=payload)
