# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Development and validation of an interpretable machine learning model for estimating hospital-acquired infection among hip fracture patients undergoing surgery: a large population-based cohort study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Age = st.sidebar.selectbox("Age", ("50-59 years", "60-69 years", "70-79 years", "80-89 years", "90-100 years", ">100 years"))
Sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
Fracture = st.sidebar.selectbox("Fracture type ", ("Femoral neck fracture", "Intertrochanteric fracture"))
Operation = st.sidebar.selectbox("Operation", ("Hip joint replacement", "Internal fixation", "Others"))
Comorbidities = st.sidebar.selectbox("Number of comorbidities", ("0", "1", "2", "3"))
Anemia = st.sidebar.selectbox("Anemia", ("No", "Yes"))
Hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
Coronarydisease = st.sidebar.selectbox("Coronary disease", ("No", "Yes"))
Cerebrovasculardisease = st.sidebar.selectbox("Cerebrovascular disease", ("No", "Yes"))
Heartfailure = st.sidebar.selectbox("Heart failure", ("No", "Yes"))
Atherosclerosis = st.sidebar.selectbox("Atherosclerosis", ("No", "Yes"))
Renalfailure = st.sidebar.selectbox("Renal failure", ("No", "Yes"))
Nephroticsyndrome = st.sidebar.selectbox("Nephrotic syndrome", ("No", "Yes"))
Respiratorysystemdisease = st.sidebar.selectbox("Respiratory system disease", ("No", "Yes"))
Gastrointestinalbleeding = st.sidebar.selectbox("Gastrointestinal bleeding", ("No", "Yes"))
Liverfailure = st.sidebar.selectbox("Liver failure", ("No", "Yes"))
Cirrhosis = st.sidebar.selectbox("Cirrhosis", ("No", "Yes"))
Diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
Dementia = st.sidebar.selectbox("Dementia", ("No", "Yes"))
Cancer = st.sidebar.selectbox("Cancer", ("No", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round-web.pkl")
    x = pd.DataFrame([[Age, Sex, Fracture, Operation, Comorbidities, Anemia, Hypertension, Coronarydisease, Cerebrovasculardisease, Heartfailure, Atherosclerosis, Renalfailure, Nephroticsyndrome, Respiratorysystemdisease, Gastrointestinalbleeding, Liverfailure, Cirrhosis, Diabetes, Dementia, Cancer]],
                     columns=["Age", "Sex", "Fracture", "Operation", "Comorbidities", "Anemia", "Hypertension", "Coronarydisease", "Cerebrovasculardisease", "Heartfailure", "Atherosclerosis", "Renalfailure", "Nephroticsyndrome", "Respiratorysystemdisease", "Gastrointestinalbleeding", "Liverfailure", "Cirrhosis", "Diabetes", "Dementia", "Cancer"])
    x = x.replace(["Male", "Female"], [1, 2])
    x = x.replace(["50-59 years", "60-69 years", "70-79 years", "80-89 years", "90-100 years", ">100 years"], [5, 6, 7, 8, 9, 10])
    x = x.replace(["Femoral neck fracture", "Intertrochanteric fracture"], [1, 2])
    x = x.replace(["Hip joint replacement", "Internal fixation", "Others"], [1, 2, 3])
    x = x.replace(["0", "1", "2", "3"], [0, 1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])


    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of hospital-acquired infection: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.023:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.023:
        st.markdown(f"Recommendations: For patients in the low-risk groups, common healthcare was recommended.")
    else:
        st.markdown(f"Recommendations: Patients in the high-risk groups were 4.56-fold chances to suffer from hospital-acquired infection than patients in the low-risk groups (P<0.001). More attentions should be paid to those patients, and prophylactic antibiotic was recommened.")

st.subheader('About the model')
st.markdown('This online calculator is freely accessible, and it’s algorithm was based on the XGBoosting machine learning model. The area under the curve of the model was 0.817 (95%CI: 0.778-0.853).')
