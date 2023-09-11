import pandas
import pandas as pd
from pandasql import sqldf
from clearml import Task,Dataset
import config
def data_enrich(df):
    data = df.copy()
    data["date"] = pd.date_range(start="01/01/2022",periods=len(data))
    return data

def data_query(query):
    try:
        data_path = "data/loan.csv"
        out_data_file = "data/loan_data_ingested.csv"
        parquet_file = "data/loan_data_ingested.parquet"
        _df = pd.read_csv(data_path)
        df = data_enrich(_df)
        loan_dataframe = df.copy()
        data_fetched = sqldf(query=query)
        data_fetched.to_csv(out_data_file,index=False)
        data_fetched.drop(columns=["Loan_Status"]).to_parquet(parquet_file)
        return data_fetched,out_data_file,parquet_file
    except:
        raise ValueError("SQL Process failed")

task = Task.init(project_name=config.PROJECT_NAME,task_name="data_ingestion",task_type="data_processing",reuse_last_task_id=False)
query_parameter = {"query":"SELECT * FROM loan_dataframe"}
task.connect(query_parameter)

#s = 'SELECT * FROM loan_dataframe WHERE strftime("%Y-%m-%d", `date`) <= strftime("%Y-%m-%d", "2022-02-01")'
load_dataset,loan_dataset_path,parquet_path = data_query(query=query_parameter["query"])

dataset = Dataset.create(dataset_name=config.INGESTED_DATASET_NAME,dataset_project=config.PROJECT_NAME)
dataset.add_files(
    loan_dataset_path
)
dataset.add_files(
    parquet_path
)
dataset.get_logger().report_table(title="Initial Dataset for Loan Application",series="head",table_plot=load_dataset.head(10))

dataset.finalize(auto_upload=True)

