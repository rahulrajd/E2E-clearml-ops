from clearml import Task,Dataset
from evidently.test_suite import TestSuite
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.test_preset import BinaryClassificationTestPreset
from clearml.automation import TriggerScheduler
from config import *
import pandas as pd

def check_drift():
    train_df = pd.read_csv("data/loan_data_ingested.csv")
    batch_df = pd.read_csv("data/ref_data_drift.csv")

    train_df = train_df.drop(["Loan_ID","Loan_Status","date"],axis=1)
    batch_df = batch_df.drop(["Loan_ID"],axis=1)

    #adding noise to simulate data drift
    batch_df[batch_df.select_dtypes(exclude="object").columns] = batch_df.select_dtypes(exclude="object") * 10

    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(reference_data=train_df, current_data=batch_df)
    if report.as_dict()["metrics"][0]["result"]["dataset_drift"]:
        task = Task.get_task(task_id="b1c94a6fbbef4b05b9079b77447b0371",project_name=PROJECT_NAME)
        cloned_task = Task.clone(source_task=task,name="re-trigger_training")
        Task.enqueue(cloned_task.id,queue_name="services")


from clearml.automation import TaskScheduler
Task.init(project_name=PROJECT_NAME,task_name="retraining_drift")
scheduler = TaskScheduler()
scheduler.add_task(
    name='recurring pipeline job',
    schedule_task_id=Task.get_task(project_name=PROJECT_NAME, task_name='retraining_drift'),
    queue='default',
    minute=1,
    recurring=True,
)
scheduler.add_task(
    name="drift_detect",
    schedule_function=check_drift,
    recurring=True,
    minute=30,
    execute_immediately=True
)

#scheduler.start_remotely(queue="triggers")
scheduler.start()
