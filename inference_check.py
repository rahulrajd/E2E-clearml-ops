import requests


sample_payload = {"Gender": ["Male"], "Married": ["Yes"], "Dependents": [0], "Education": ["Graduate"], "Self_Employed": ["Yes"], "ApplicantIncome": [0], "CoapplicantIncome": [0], "LoanAmount": [0], "Loan_Amount_Term": [24], "Credit_History": [1], "Property_Area": ["Urban"]}


response = requests.post(url='http://127.0.0.1:8080/serve/loan_application',headers={'accept': 'application/json', 'Content-Type': 'application/json'},json=sample_payload)

print(response.json())