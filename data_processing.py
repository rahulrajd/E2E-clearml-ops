import os.path
from pathlib import Path
import pandas as pd
from clearml import Dataset, Task
import ydata_profiling
import config
import category_encoders as ce
from sklearn.impute import SimpleImputer
from joblib import dump
task = Task.init(
    project_name=config.PROJECT_NAME,
    task_name='preprocess_data',
    task_type='data_processing',
    reuse_last_task_id=False
)

dataset = Dataset.get(
    dataset_project=config.PROJECT_NAME,
    dataset_name=config.INGESTED_DATASET_NAME,
    alias="ingested_data for data preprocessing"
)
data_file = dataset.get_local_copy()
print(f"Using dataset ID: {dataset.id}")

loan_data = pd.read_csv(data_file+"/loan_data_ingested.csv")
preprocessed_dataset = Dataset.create(
    dataset_name="preprocessed_data",
    dataset_project=config.PROJECT_NAME,
    parent_datasets=[dataset.id]
)

#EDA process
loan_data.drop(columns=["Loan_ID","date"],inplace=True)
print(loan_data.describe())
preprocessed_dataset.get_logger().report_table(title="Description of Data",series="values",table_plot=loan_data.describe())

#Preprocess

loan_data_to_preprocess = loan_data.drop(columns=["Loan_Status"])
loan_data_to_preprocess["Dependents"] = loan_data_to_preprocess["Dependents"].replace("3+","3")
loan_data_to_preprocess["Dependents"] = loan_data_to_preprocess["Dependents"].fillna("0").astype("int64")

category_columns = loan_data_to_preprocess[loan_data_to_preprocess.select_dtypes(include=["object"]).columns]
category_columns = category_columns.fillna("Null")

numerical_columns = loan_data_to_preprocess[loan_data_to_preprocess.select_dtypes(exclude=["object"]).columns]
numerical_columns = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(numerical_columns),columns=numerical_columns.columns)

encoder = ce.OneHotEncoder()
category_df = encoder.fit_transform(category_columns)

target_encoding_df = loan_data['Loan_Status'].map({"Y":1,"N":0})
final_dataset = pd.DataFrame(pd.concat([category_df,numerical_columns,target_encoding_df],axis=1))
final_dataset.to_csv("data/processed_data.csv",index=False)
dump(encoder,"data/encoder.joblib")
task.upload_artifact(name='categorical_encoder', artifact_object=encoder)
task.upload_artifact(name='preprocessed_data',artifact_object=final_dataset)

preprocessed_dataset.add_files("data/processed_data.csv")
preprocessed_dataset.finalize(auto_upload=True)
print(f"Created preprocessed dataset with ID: {preprocessed_dataset.id}")
