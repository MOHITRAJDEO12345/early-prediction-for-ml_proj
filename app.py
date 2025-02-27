import os
import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st
import seaborn as sns
from streamlit_option_menu import option_menu
import time
import matplotlib.pyplot as plt
import json
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline

# Set page config with icon
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

diabetes_model = pickle.load(open('diabetes/diabetes_model.sav', 'rb'))
# asthama_model = pickle.load(open('asthama/model.pkl', 'rb'))

import joblib
asthama_model = joblib.load("asthama/model.pkl")

cardio_model = pickle.load(open('cardio_vascular/xgboost_cardiovascular_model.pkl', 'rb'))

# stroke_model = pickle.load(open('stroke/stroke_model.sav', 'rb'))

stroke_model = joblib.load("stroke/finalized_model.pkl")


prep_asthama = pickle.load(open('asthama/preprocessor.pkl', 'rb'))

with st.sidebar:
    st.title("🩺 Disease Prediction")
    
    selected = option_menu(
        menu_title="Navigation",
        options=['Home','Checkbox-to-disease-predictor',  'AI Health Consultant', 'Mental-Analysis', 'Diabetes Prediction', 'Asthma Prediction', 'Cardiovascular Disease Prediction', 'Stroke Prediction', 'Data Visualization' ],
        icons=['house', 'activity', 'lungs', 'heart-pulse', 'brain', 'bar-chart', 'chat'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#111111"},  # Darker background
            "icon": {"color": "#FF0000", "font-size": "20px"},  # Red icons
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},  # White text
            "nav-link-selected": {"background-color": "#FF0000", "color": "#FFFFFF"},
        },
    )

# Utility function to safely convert input to float
def safe_float(value, default=0.0):
    try:
        return float(value)
    except ValueError:
        return default  # Assigns default value if conversion fails


# 🚀 Home Page
if selected == 'Home':
    st.title("🩺 AI-Powered Health & Lifestyle Disease Prediction")

    st.markdown("""
    ## Welcome to the **AI-Powered Health Prediction System**!  
    This tool provides **early prediction and analysis** for various health conditions using **Machine Learning & NLP**.
    
    ### 🏥 Available Features:
    - **✅ Checkbox-based Lifestyle Disease Predictor** using **BiomedNLP-PubMedBERT**  
    - **🤖 AI Chatbot for Health Assistance** (Ask health-related questions)  
    - **🧠 Mental Health Assessment**  
    - **🩸 Disease Predictors**:
      - Diabetes  
      - Asthma  
      - Stroke  
      - Cardiovascular Disease  
    - **📊 Data Visualizer** (Analyze trends in health conditions)  
                
    👉 Select an option from the sidebar to proceed!  
    """)

    with st.expander("🚀 Quick Start Guide"):
        st.write("""
        1. Select a **health prediction model** from the sidebar.
        2. Enter your details in the input fields.
        3. Click **Predict** to get your result.
        4. View personalized **health insights & recommendations**.
        """)


    # Disclaimer Section
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:** This application has been developed using **real-world healthcare datasets** sourced from Kaggle:  
    - [Stroke Prediction Dataset](http://kaggle.com/code/chanchal24/stroke-prediction-using-python/input?select=healthcare-dataset-stroke-data.csv)  
    - [Asthma Analysis & Prediction](https://www.kaggle.com/code/bryamblasrimac/asthma-eda-prediction-f2score-85/input)  
    - [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)  

    The predictions are generated using **machine learning models** trained on these datasets, incorporating **evaluation metrics and graphical insights** to enhance interpretability.  

    However, this tool has **not undergone clinical validation** and should be used **for informational and educational purposes only**. It is not intended to serve as a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare provider for medical advice.
    """)

if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML (SVC)')
    st.image("https://cdn-icons-png.flaticon.com/512/2919/2919950.png", width=100)

    st.markdown("""
    This model predicts the likelihood of **Diabetes** based on various health parameters.  
    Please enter the required medical details below and click **"Diabetes Test Result"** to get the prediction.
    """)



    # Create columns for better input organization
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = safe_float(st.text_input("Number of Pregnancies", "0"))
        Glucose = safe_float(st.text_input("Glucose Level", "100"))
        BloodPressure = safe_float(st.text_input("Blood Pressure", "80"))
        SkinThickness = safe_float(st.text_input("Skin Thickness", "20"))

    with col2:
        Insulin = safe_float(st.text_input("Insulin Level", "79"))
        BMI = safe_float(st.text_input("BMI (Body Mass Index)", "25.0"))
        DiabetesPedigreeFunction = safe_float(st.text_input("Diabetes Pedigree Function", "0.5"))
        Age = st.number_input("Enter Age", min_value=10, max_value=100, value=30, step=1)

    with col1:
        if st.button('Diabetes Test Result'):
            try:
                input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
                
                with st.spinner("⏳ Predicting... Please wait..."):
                    time.sleep(2)  # Simulating delay (remove in actual use)
                    diab_prediction = diabetes_model.predict(input_data)

                
                
                result = "🛑 The person is diabetic" if diab_prediction[0] == 1 else "✅ The person is not diabetic"
                if diab_prediction[0] == 0:
                    st.balloons()  # Or use st.confetti() if you install the library
                st.success(result)

            except Exception as e:
                st.error(f"❌ Error: {e}")


if selected == 'Asthma Prediction':
    st.title('🌬️ Asthma Prediction using ML')
    st.image("https://cdn-icons-png.flaticon.com/512/3462/3462191.png", width=100)

    st.markdown("""
    This model predicts the likelihood of **Asthma** based on health factors.  
    Enter your details and click **"Asthma Test Result"** to get the prediction.
    """)

    col1, col2 = st.columns(2)

    with col1:
        Gender_Male = st.radio("Gender", ["Female", "Male"])
        Gender_Male = 1 if Gender_Male == "Male" else 0

        Smoking_Status = st.radio("Smoking Status", ["Non-Smoker", "Ex-Smoker"])
        Smoking_Status_Ex_Smoker = 1 if Smoking_Status == "Ex-Smoker" else 0
        Smoking_Status_Non_Smoker = 1 if Smoking_Status == "Non-Smoker" else 0

    with col2:
        Age = st.slider("Enter Age (Normalized)", min_value=0.0, max_value=0.914894, value=0.5)
        Peak_Flow = st.slider("Peak Flow (L/sec)", min_value=0.1, max_value=1.0, value=0.5)

    with col1:
        if st.button('Asthma Test Result'):
            try:
                # Prepare raw input
                raw_input = np.array([[Gender_Male, Smoking_Status_Ex_Smoker, Smoking_Status_Non_Smoker, Age, Peak_Flow]])

                # Check if preprocessing is needed
                if prep_asthama is not None and hasattr(prep_asthama, "transform"):
                    processed_input = prep_asthama.transform(raw_input)  # Use transform if prep_asthama exists
                else:
                    processed_input = raw_input  # If no scaler, use raw input

                with st.spinner("⏳ Predicting... Please wait..."):
                    time.sleep(2)  # Simulating delay (remove in actual use)
                    asthma_prediction = asthama_model.predict(processed_input)

                result = "🛑 High risk of asthma" if asthma_prediction[0] == 1 else "✅ Low risk of asthma"
                if asthma_prediction[0] == 0:
                    st.balloons()
                st.success(result)

            except Exception as e:
                st.error(f"❌ Error: {e}")


if selected == 'Cardiovascular Disease Prediction':
    st.title('❤️ Cardiovascular Disease Prediction')
    st.image("https://cdn-icons-png.flaticon.com/512/2919/2919950.png", width=100)

    st.markdown("""
    This model predicts the likelihood of **Cardiovascular Disease** based on various health parameters.  
    Please enter the required medical details below and click **"Cardio Test Result"** to get the prediction.
    """)

    # Input Fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=29, max_value=64, value=40, step=1)
        ap_hi = st.slider("Systolic Blood Pressure (ap_hi)", min_value=90, max_value=180, value=120)
        ap_lo = st.slider("Diastolic Blood Pressure (ap_lo)", min_value=60, max_value=120, value=80)
        weight = st.number_input("Weight (kg)", min_value=40.0, max_value=180.0, value=70.0, step=0.1)

    with col2:
        cholesterol = st.radio("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
        cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]

        gluc = st.radio("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
        gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]

        smoke = st.radio("Smoking Status", ["No", "Yes"])
        smoke = 1 if smoke == "Yes" else 0

        alco = st.radio("Alcohol Consumption", ["No", "Yes"])
        alco = 1 if alco == "Yes" else 0

        active = st.radio("Physically Active", ["No", "Yes"])
        active = 1 if active == "Yes" else 0

    # Prediction Button
    if st.button('Cardio Test Result'):
        try:
            # Preparing Input Data
            input_data = np.array([[age, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, weight]])

            with st.spinner("⏳ Predicting... Please wait..."):
                time.sleep(2)  # Simulating Model Inference
                cardio_prediction = cardio_model.predict(input_data)

            # Display Result
            result = "🛑 High risk of cardiovascular disease" if cardio_prediction[0] == 1 else "✅ Low risk of cardiovascular disease"
            if cardio_prediction[0] == 0:
                st.balloons()
            st.success(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")





if selected == 'Stroke Prediction':
    st.title('🧠 Stroke Prediction using ML')
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209265.png", width=100)

    st.markdown("""
    This model predicts the likelihood of **Stroke** based on various health factors.  
    Enter your details and click **"Stroke Test Result"** to get the prediction.
    """)

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=0, max_value=82, value=50, step=1)
        Hypertension = st.radio("Hypertension", ["No", "Yes"])
        Hypertension = 1 if Hypertension == "Yes" else 0

        Heart_Disease = st.radio("Heart Disease", ["No", "Yes"])
        Heart_Disease = 1 if Heart_Disease == "Yes" else 0

    with col2:
        Ever_Married = st.radio("Ever Married", ["No", "Yes"])
        Ever_Married = 1 if Ever_Married == "Yes" else 0

        Avg_Glucose_Level = st.slider("Average Glucose Level", min_value=55.23, max_value=267.61, value=120.0)
        BMI = st.slider("BMI", min_value=13.5, max_value=97.6, value=25.0)

        Smoking_Status = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Smokes", "Unknown"])
        Smoking_Status = {"Never Smoked": 0, "Former Smoker": 1, "Smokes": 2, "Unknown": 3}[Smoking_Status]

    with col1:
        if st.button('Stroke Test Result'):
            try:
                input_data = np.array([[Age, Hypertension, Heart_Disease, Ever_Married, Avg_Glucose_Level, BMI, Smoking_Status]])

                with st.spinner("⏳ Predicting... Please wait..."):
                    time.sleep(2)
                    stroke_prediction = stroke_model.predict(input_data)

                result = "🛑 High risk of stroke" if stroke_prediction[0] == 1 else "✅ Low risk of stroke"
                if stroke_prediction[0] == 0:
                    st.balloons()
                st.success(result)

            except Exception as e:
                st.error(f"❌ Error: {e}")




if selected == 'Data Visualization':
    # st.set_page_config(page_title="Data Visualizer",
    #                 page_icon="📊", layout="centered")
    st.title(" 📊 Data Visualization")

    working_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = f"{working_dir}/data_csv"

    files_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    selected_file = st.selectbox("Select a file", files_list, index=None)

    if selected_file:

        file_path = os.path.join(folder_path, selected_file)

        df = pd.read_csv(file_path)

        columns = df.columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            st.write("")
            st.write(df.head())

        with col2:
            x_axis = st.selectbox("Select X-axis", options=columns + ["None"])
            y_axis = st.selectbox("Select Y-axis", options=columns + ["None"])

            plot_list = ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot", "Distribution Plot", "Count Plot", "Pair Plot"]

            selected_plot = st.selectbox("Select a plot", options=plot_list, index=None)

            # st.write(x_axis)
            # st.write(y_axis)
            # st.write(selected_plot)

        if st.button("Generate Plot"):

            fig, ax = plt.subplots(figsize=(6,4))

            if selected_plot == "Line Plot":
                sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)

            elif selected_plot == "Bar Plot":
                sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
            
            elif selected_plot == "Scatter Plot":
                sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
            
            elif selected_plot == "Histogram":
                sns.histplot(df[x_axis], ax=ax)
            
            elif selected_plot == "Box Plot":
                sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)

            elif selected_plot == "Distribution Plot":
                sns.kdeplot(df[x_axis], ax=ax)
            
            elif selected_plot == "Count Plot":
                sns.countplot(x=x_axis, data=df, ax=ax)
            
            elif selected_plot == "Pair Plot":
                sns.pairplot(df, ax=ax)

            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

            plt.title(f"{selected_plot} of {x_axis} vs {y_axis}", fontsize=12)
            plt.xlabel(x_axis, fontsize=10)
            plt.ylabel(y_axis, fontsize=10)

            st.pyplot(fig)


if selected == 'AI Health Consultant':
    st.title("🩺 AI Health Consultation Assistant")
    st.markdown("### Discuss Your Health Concerns with Our AI-powered Chatbot")
    st.write("Ask about **Diabetes, Asthma, Stroke, Cardiovascular Disease, or Mental Health.**")

    genai.configure(api_key="AIzaSyD9x7Kz8adDo6-nVyk9MAQjlwD4lTeKc84")

    # Custom Styling
    st.markdown("""
        <style>
            .prompt-box { 
                background-color: #000000; 
                padding: 12px; 
                border-radius: 8px; 
                font-size: 14px; 
                font-family: sans-serif;
                margin-bottom: 10px;
                border: 1px solid #dee2e6;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 💡 Common Health Queries")

    prompt_options = [
        ("Diabetes – Diet", "What foods should I eat if I have diabetes?"),
        ("Diabetes – Exercise", "What type of workouts help control blood sugar levels?"),
        ("Asthma – Triggers", "What are common asthma triggers?"),
        ("Asthma – Treatment", "What are the best medications for asthma?"),
        ("Stroke – Symptoms", "What are the early warning signs of a stroke?"),
        ("Stroke – Prevention", "How can I reduce my risk of stroke?"),
        ("Cardiovascular – Heart Health", "How can I reduce my risk of heart disease?"),
        ("Cardiovascular – Blood Pressure", "What lifestyle changes can lower high blood pressure?"),
        ("Mental Health – Stress Management", "How can I manage stress effectively?"),
        ("Mental Health – Sleep Disorders", "What are the causes and treatments for sleep disorders?")
    ]

    # Display prompts in two columns (2 prompts per row)
    cols = st.columns(2)
    for i in range(0, len(prompt_options), 2):
        with cols[0]: 
            if i < len(prompt_options):
                label, prompt = prompt_options[i]
                st.markdown(f"""<div class="prompt-box"><strong>{label}</strong><br>{prompt}</div>""", unsafe_allow_html=True)

        with cols[1]: 
            if i+1 < len(prompt_options):
                label, prompt = prompt_options[i+1]
                st.markdown(f"""<div class="prompt-box"><strong>{label}</strong><br>{prompt}</div>""", unsafe_allow_html=True)

    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input field
    user_prompt = st.chat_input("Ask about Diabetes, Asthma, Stroke, Cardiovascular Disease, or Mental Health...")

    # List of allowed topics
    allowed_keywords = ["diabetes", "asthma", "stroke", "cardiovascular", "heart", "blood pressure", 
                        "mental health", "depression", "stress", "cholesterol", "sleep disorders"]

    if user_prompt:
        # Display user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Restriction: Only process if related to health topics
        if any(keyword in user_prompt.lower() for keyword in allowed_keywords):
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(user_prompt)

            if response and hasattr(response, "text"):
                assistant_response = response.text
            else:
                assistant_response = "I'm sorry, I couldn't generate a response."

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

            # Display assistant's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        else:
            # Restriction message
            restriction_msg = "**⚠️ This chatbot only responds to health-related topics.**\nPlease ask about Diabetes, Asthma, Stroke, Cardiovascular Disease, or Mental Health."
            st.session_state.chat_history.append({"role": "assistant", "content": restriction_msg})
            
            with st.chat_message("assistant"):
                st.markdown(restriction_msg)


# if selected == 'Text-to-disease-predictor':
#     st.title("🔮 Text-to-Disease Predictor")
#     st.markdown("### Enter your symptoms to predict the likelihood of a disease!")
#     st.write("Try entering symptoms like 'I have a fever and cough'.")
#     st.write("This tool uses a pre-trained model to predict the likelihood of common diseases based on your symptoms.")

#     # Load the pre-trained model
#     classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#     # Streamlit UI
#     st.title("Text-Based Symptom Analysis")
#     user_input = st.text_area("Enter your symptoms (e.g., 'I have a fever and cough'):")

#     # Define candidate diseases
#     candidate_labels = [
#     "Diabetes", "Hypertension", "Obesity", "Cardiovascular Disease", "COPD",
#     "Liver Disease", "Kidney Disease", "Metabolic Syndrome", "Osteoarthritis",
#     "GERD", "Cancer", "Alzheimer's Disease", "Depression", "Sleep Apnea",
#     "Thyroid Disorders"
#     ]

#     with st.expander("Click to view common lifestyle diseases"):
#         diseases = [
#             "Diabetes", "Hypertension", "Obesity", "Cardiovascular Disease", "COPD",
#             "Liver Disease", "Kidney Disease", "Metabolic Syndrome", "Osteoarthritis",
#             "GERD", "Cancer", "Alzheimer's Disease", "Depression", "Sleep Apnea",
#             "Thyroid Disorders"
#         ]
        
#         for disease in diseases:
#             st.write(f"- {disease}")

#     with st.expander("Click to view common lifestyle diseases and their symptoms"):
#         diseases = {
#             "Diabetes": ["Frequent urination", "Increased thirst", "Unexplained weight loss", "Fatigue", "Blurred vision"],
#             "Hypertension": ["Headache", "Dizziness", "Chest pain", "Shortness of breath", "Nosebleeds"],
#             "Obesity": ["Excess body fat", "Breathlessness", "Joint pain", "Increased sweating", "Low energy levels"],
#             "Cardiovascular Disease": ["Chest pain", "Shortness of breath", "Dizziness", "Irregular heartbeat", "Fatigue"],
#             "COPD": ["Chronic cough", "Shortness of breath", "Wheezing", "Chest tightness", "Frequent respiratory infections"],
#             "Liver Disease": ["Jaundice", "Abdominal pain", "Swelling in legs", "Chronic fatigue", "Nausea"],
#             "Kidney Disease": ["Swelling in legs", "Fatigue", "Loss of appetite", "Changes in urination", "Muscle cramps"],
#             "Metabolic Syndrome": ["High blood sugar", "High blood pressure", "Increased waist size", "High cholesterol", "Fatigue"],
#             "Osteoarthritis": ["Joint pain", "Stiffness", "Swelling", "Reduced flexibility", "Bone spurs"],
#             "GERD": ["Heartburn", "Acid reflux", "Difficulty swallowing", "Chronic cough", "Sore throat"],
#             "Cancer": ["Unexplained weight loss", "Persistent cough", "Fatigue", "Lumps", "Skin changes"],
#             "Alzheimer's Disease": ["Memory loss", "Confusion", "Difficulty in problem-solving", "Mood changes", "Disorientation"],
#             "Depression": ["Persistent sadness", "Loss of interest", "Sleep disturbances", "Fatigue", "Difficulty concentrating"],
#             "Sleep Apnea": ["Loud snoring", "Pauses in breathing", "Daytime drowsiness", "Morning headaches", "Irritability"],
#             "Thyroid Disorders": ["Weight changes", "Fatigue", "Hair loss", "Mood swings", "Temperature sensitivity"]
#         }

#         for disease, symptoms in diseases.items():
#             st.markdown(f"### {disease}")
#             st.write("**Common Symptoms:**")
#             for symptom in symptoms:
#                 st.write(f"- {symptom}")
#             st.write("---")  # Adds a separator between diseases for better readability

#     if st.button("Predict Disease"):
#         if user_input:
#             result = classifier(user_input, candidate_labels)
#             st.write("Possible Conditions:")
#             for disease, score in zip(result["labels"], result["scores"]):
#                 st.write(f"🩺 {disease}: {round(score * 100, 2)}% risk")



if selected == 'Checkbox-to-disease-predictor':
# Load transformer model
    classifier = pipeline("zero-shot-classification", model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    # Define symptoms for each disease
    diseases = {
        "Diabetes": ["Frequent urination", "Increased thirst", "Unexplained weight loss", "Fatigue", "Blurred vision"],
        "Hypertension": ["Headache", "Dizziness", "Chest pain", "Shortness of breath", "Nosebleeds"],
        "Obesity": ["Excess body fat", "Breathlessness", "Joint pain", "Increased sweating", "Low energy levels"],
        "Cardiovascular Disease": ["Chest pain", "Shortness of breath", "Dizziness", "Irregular heartbeat", "Fatigue"],
        "COPD": ["Chronic cough", "Shortness of breath", "Wheezing", "Chest tightness", "Frequent respiratory infections"],
        "Liver Disease": ["Jaundice", "Abdominal pain", "Swelling in legs", "Chronic fatigue", "Nausea"],
        "Kidney Disease": ["Swelling in legs", "Fatigue", "Loss of appetite", "Changes in urination", "Muscle cramps"],
        "Metabolic Syndrome": ["High blood sugar", "High blood pressure", "Increased waist size", "High cholesterol", "Fatigue"],
        "Osteoarthritis": ["Joint pain", "Stiffness", "Swelling", "Reduced flexibility", "Bone spurs"],
        "Gastroesophageal Reflux Disease": ["Heartburn", "Acid reflux", "Difficulty swallowing", "Chronic cough", "Sore throat"],
        "Depression": ["Persistent sadness", "Loss of interest", "Sleep disturbances", "Fatigue", "Difficulty concentrating"],
        "Sleep Apnea": ["Loud snoring", "Pauses in breathing", "Daytime drowsiness", "Morning headaches", "Irritability"],
    }

    # Streamlit UI
    st.title("🩺 Hybrid Symptom Checker")
    st.write("Select your symptoms and get AI-powered predictions!")

    selected_symptoms = []

    # Create symptom selection with markdown separation and three columns
    disease_keys = list(diseases.keys())

    for i in range(0, len(disease_keys), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(disease_keys):
                disease = disease_keys[i + j]
                with cols[j]:
                    st.markdown(f"### {disease}")
                    for symptom in diseases[disease]:
                        if st.checkbox(symptom, key=f"{disease}_{symptom}"):
                            selected_symptoms.append(symptom)

    if st.button("🔍 Predict Disease"):
        if selected_symptoms:
            user_input = ", ".join(selected_symptoms)  # Convert symptoms to text

            # 1️⃣ Custom Symptom Matching Approach
            disease_scores = {disease: 0 for disease in diseases.keys()}
            for disease, symptoms in diseases.items():
                matches = sum(symptom in selected_symptoms for symptom in symptoms)
                disease_scores[disease] = matches / len(symptoms)  # Normalize by symptom count

            # Normalize to percentage
            symptom_match_scores = {d: round(score * 100, 2) for d, score in disease_scores.items()}

            # 2️⃣ AI Model Prediction
            ai_results = classifier(user_input, list(diseases.keys()))
            ai_scores = {ai_results["labels"][i]: round(ai_results["scores"][i] * 100, 2) for i in range(len(ai_results["labels"]))}

            # 3️⃣ Hybrid Score Calculation (Average of Both Scores)
            final_scores = {}
            for disease in diseases.keys():
                symptom_score = symptom_match_scores.get(disease, 0)
                ai_score = ai_scores.get(disease, 0)
                final_scores[disease] = round((symptom_score + ai_score) / 2, 2)  # Averaging

            # Sort by final score
            sorted_final_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

            # Display results
            st.write("### 🔬 Possible Conditions (Hybrid Model Prediction):")
            for disease, score in sorted_final_scores:
                if score > 0:
                    st.write(f"🩺 {disease}: {score}% match")
        else:
            st.write("⚠️ Please select at least one symptom.")


if selected == "Mental-Analysis":
    # Load the Hugging Face model
    classifier = pipeline("text-classification", model="mental/mental-roberta-base")

    # Sidebar with title and markdown
    st.sidebar.title("🧠 Mental Health Analysis")
    st.sidebar.markdown("""
    Analyze mental health symptoms using a **pre-trained AI model**.  
    This tool predicts **Depression, Anxiety, PTSD, Bipolar Disorder, and Schizophrenia** based on text input.
    """)

    # Main content
    st.title("🔬 Mental Health Text Analysis")
    st.markdown("Enter a description of your mental state, and the AI will predict possible conditions.")

    # User input
    user_input = st.text_area("Describe your symptoms (e.g., 'I feel hopeless and anxious all the time.'):")

    if st.button("Analyze"):
        if user_input:
            # Get predictions
            results = classifier(user_input)

            # Extract labels and scores
            labels = [res["label"] for res in results]
            scores = [res["score"] for res in results]

            # Display results
            st.write("### Predictions:")
            for label, score in zip(labels, scores):
                st.write(f"🩺 **{label}**: {round(score * 100, 2)}% confidence")

            # Create a bar chart
            fig, ax = plt.subplots()
            ax.barh(labels, scores, color=['blue', 'red', 'green', 'purple', 'orange'])
            ax.set_xlabel("Confidence Score")
            ax.set_title("Mental Health Analysis Results")
            ax.set_xlim(0, 1)  # Ensure scores are between 0 and 1
            st.pyplot(fig)

