import requests
from collections import OrderedDict
import streamlit as st
import json

st.set_page_config(layout="wide")
# Application
st.title("Loan Application")

def get_loan_request():
    Gender = st.sidebar.radio("Select your Gender",options=["Male","Female"])
    Married = st.sidebar.radio("Are you married", options=["Yes", "No"])
    Dependents = st.sidebar.slider("No of Dependents", 0, 10, 0)
    ApplicantIncome = st.sidebar.slider("Applicant Income", 0, 9999, 0)
    CoapplicantIncome = st.sidebar.slider("Co-Applicant Income", 0, 9999, 0)
    Education = st.sidebar.radio(
        "Are you a Graduate ?",
        [
            "Graduate",
            "Not Graduate",
        ],
    )

    LoanAmount = st.sidebar.slider("Loan amount", 0, 9999, 0)
    Loan_Amount_Term= st.sidebar.slider("Preferred Tenure in weeks", 0, 500, 24, step=1)
    Property_Area = st.sidebar.radio("your Residence property",options=["Urban","Semiurban","Rural"])
    Self_Employed = st.sidebar.radio("Self Employed", options=["Yes", "No"])
    Credit_History = st.sidebar.radio("Self attested credit history", options=[1, 0])
    submit = st.sidebar.button("Submit")
    if submit:

        payload = OrderedDict(
        {
            "Gender":[Gender],
            "Married": [Married],
            "Dependents": [Dependents],
            "Education": [Education],
            "Self_Employed": [Self_Employed],
            "ApplicantIncome": [ApplicantIncome],
            "CoapplicantIncome": [CoapplicantIncome],
            "LoanAmount": [LoanAmount],
            "Loan_Amount_Term": [Loan_Amount_Term],
            "Credit_History": [Credit_History],
            "Property_Area": [Property_Area],
        }

    )
        try:
            response = requests.post("http://127.0.0.1:5010/check_payload/", data=json.dumps(payload))
            result = response.json()["prediction"]
            #result, shap_values,data_row = response.json()["prediction"],response.json()["shap"],response.json()["data_row"]
            st.header("Application Status (model prediction):")
            if result == 1:
                st.success("Your loan has been approved!")
            elif result == 0:
                st.error("Your loan has been rejected!")

        except:
            raise ConnectionError



# Input Side Bar
st.sidebar.header("User input:")
loan_request = get_loan_request()

#SHAP Explainer Dashboard
#parquet_dataset_path = Dataset.get(dataset_project=config.PROJECT_NAME,dataset_name=config.INGESTED_DATASET_NAME).get_local_copy()
#model_task = Task.get_task(project_name=config.PROJECT_NAME,task_name="model_training")


#st.header("Feature Importance")
#left, mid, right = st.columns(3)
#with left:
#    plt.title("Feature importance based on SHAP values")
#    shap.summary_plot(shap_values[1], data_row)
#    st.set_option("deprecation.showPyplotGlobalUse", False)
#    st.pyplot(bbox_inches="tight")
#    st.write("---")

#with mid:
#    plt.title("Feature importance based on SHAP values (Bar)")
#    shap.summary_plot(shap_values, data_row, plot_type="bar")
#    st.pyplot(bbox_inches="tight")

