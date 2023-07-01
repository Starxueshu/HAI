# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Development and validation of a web-based interpretable machine learning application for estimating hospital-acquired infection among elderly hip fracture patients: a nationwide observational cohort study")
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
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
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
    if prediction < 0.526:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.526:
        st.markdown(f"For high-risk patients with elderly hip fractures, it is crucial to implement comprehensive hospital management measures to minimize the risk of hospital-acquired infections (HAIs). The following strategies should be considered: (1) Strict surgical protocols: Adhering to strict surgical protocols, including proper hand hygiene, aseptic techniques, and sterile equipment, can significantly reduce the risk of HAIs. Surgeons should follow infection control guidelines and ensure a clean surgical environment. (2) Prophylactic antibiotic use: Administering prophylactic antibiotics before surgery can help prevent surgical site infections. The choice of antibiotics should be based on local antimicrobial resistance patterns and guidelines to ensure optimal effectiveness. (3) Patient factors: High-risk patients should be carefully assessed for comorbidities and underlying medical conditions that may increase their susceptibility to infections.")
    else:
        st.markdown(f"While low-risk elderly hip fracture patients have a lower risk of HAIs, it is still important to implement appropriate hospital management measures to maintain patient safety. The following measures should be considered: (1) Surgical protocols: Even for low-risk patients, adherence to surgical protocols, including proper hand hygiene and aseptic techniques, is essential to minimize the risk of infections. Maintaining a clean surgical environment and following infection control guidelines are crucial. (2) Infection prevention measures: While the risk of HAIs is lower for low-risk patients, it is still important to implement infection prevention measures. This includes regular hand hygiene, proper wound care, and appropriate isolation precautions when necessary.")

st.subheader('About the model')
st.markdown('This study developed and validated a web-based interpretable machine learning application for estimating the risk of hospital-acquired infection (HAI) among elderly hip fracture patients. This nationwide observational cohort study analyzed data from 93,216 patients. The XGBoosting model performed best in terms of accuracy, precision, and F1 score. The online application, based on the optimal model, provides predicted HAI risk, risk classification, and therapeutic recommendations for individual patients. This application serves as a valuable tool for doctors in stratifying patients at risk of developing HAI. This online application is freely accessible. This online calculator is freely accessible.')
