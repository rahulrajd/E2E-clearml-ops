import matplotlib.pyplot as plt
import pandas as pd
from pathlib2 import Path
from clearml import Dataset, Task
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import config
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score
task = Task.init(
    project_name=config.PROJECT_NAME,
    task_name='model_training',
    task_type="training",
    output_uri=True
)

task.set_base_docker(docker_image="python:3.10")

# Training args
model_parameters = {
    'num_estimators': 100,
    'max_depth': 10
}
task.connect(model_parameters)

data_path = Dataset.get(
    dataset_name='preprocessed_data',
    dataset_project=config.PROJECT_NAME
).get_local_copy()

#local_path = Path(data_path)
processed_training_data = pd.read_csv(data_path+"/processed_data.csv")

X= processed_training_data.drop(columns=["Loan_Status"])
y = processed_training_data["Loan_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf_clf = RandomForestClassifier(n_estimators=model_parameters["num_estimators"],max_depth=model_parameters["max_depth"])
rf_clf.fit(X_train,y_train)
y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)

print("Train Recall Score ", recall)
print("Train Accuracy ", accuracy)
task.get_logger().report_single_value(
    name='Accuracy',
    value=accuracy,

)
task.get_logger().report_single_value(
    name='Recall',
    value=recall)
dump(rf_clf,"model.joblib")


print("Done")

