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

# sleep_model = pickle.load(open('sleep_health/best_model.pkl', 'rb'))
# scaler = pickle.load(open('sleep_health/scaler.pkl', 'rb'))
# label_encoder = pickle.load(open('sleep_health/label_encoders.pkl', 'rb'))

# At the beginning of your app, when loading models:
try:
    sleep_model = pickle.load(open('sleep_health/svc_model.pkl', 'rb'))
    scaler = pickle.load(open('sleep_health/scaler.pkl', 'rb'))
    label_encoder = pickle.load(open('sleep_health/label_encoders.pkl', 'rb'))
    
    # Store the expected feature names if available
    # This depends on how your model was saved/trained
    # if hasattr(sleep_model, 'feature_names_in_'):
    #     expected_features = sleep_model.feature_names_in_
    # else:
    #     # Try to load feature names from a separate file
    #     try:
    #         with open('sleep_health/feature_names.pkl', 'rb') as f:
    #             expected_features = pickle.load(f)
    #     except:
    #         st.warning("Warning: Feature names not found. Predictions may be inaccurate.")
    #         expected_features = None
            
except FileNotFoundError:
    st.error("Error: Model files not found. Please upload the model files.")
    st.stop()

with st.sidebar:
    st.title("🩺 Disease Prediction")
    
    selected = option_menu(
        menu_title="Navigation",
        options=['Home',  'Diabetes Prediction','Hypertension Prediction', 'Cardiovascular Disease Prediction', 'Stroke Prediction','Asthma Prediction', 'Sleep Health Analysis','Mental-Analysis','Medical Consultant', 'Data Visualization'],
        icons=['house', 'activity', 'lungs', 'heart-pulse', 'brain', 'bar-chart', 'chat'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#111111"},  # Darker background
            "icon": {"color": "#FF0000", "font-size": "20px"},  # Red icons
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},  # White text
            "nav-link-selected": {"background-color": "#FF0000", "color": "#FFFFFF"},
        }
    )


# 'Checkbox-to-disease-predictor', 
# 'Text-based Disease Prediction', 
# Utility function to safely convert input to float
def safe_float(value, default=0.0):
    try:
        return float(value)
    except ValueError:
        return default  # Assigns default value if conversion fails


# 🚀 Home Page
if selected == 'Home':
    st.title("🩺 Early Prediction of Health & Lifestyle Diseases")

    st.markdown("""
    ## Welcome to the **Early Prediction of Health & Lifestyle Diseases**!  
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
    # st.markdown("""
    # **⚠️ Disclaimer:** This application has been developed using **real-world healthcare datasets** sourced from Kaggle:  

    # - [Stroke Prediction Dataset](http://kaggle.com/code/chanchal24/stroke-prediction-using-python/input?select=healthcare-dataset-stroke-data.csv)  
    # - [Asthma Analysis & Prediction](https://www.kaggle.com/code/bryamblasrimac/asthma-eda-prediction-f2score-85/input)  
    # - [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)  
    # - [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)  
    # - [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)  
    # - [Sleep Health Analysis](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
    # \


    st.markdown("""
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
                    # st.balloons()  # Or use st.confetti() if you install the library
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
        # Use actual age as input instead of normalized value
        actual_age = st.slider("Age", min_value=18, max_value=85, value=40, help="Your actual age in years")
        
        # Convert actual age to normalized value (0.0 to 0.914894)
        # Assuming normalization was done with min_age=18 and max_age=90
        min_age = 18
        max_age = 90
        Age = (actual_age - min_age) / (max_age - min_age)
        
        # Show the normalized value for reference (can be hidden in final version)
        st.info(f"Normalized age value (used by model): {Age:.6f}")
        
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
                    # st.balloons()
                    st.success(result)
                else:
                    st.error(result)
                
                # Add risk factor analysis
                st.subheader("Risk Factor Analysis")
                risk_factors = []
                
                if actual_age > 60:
                    risk_factors.append("⚠️ Age is a risk factor for asthma")
                if Smoking_Status == "Ex-Smoker":
                    risk_factors.append("⚠️ Smoking history increases asthma risk")
                if Peak_Flow < 0.5:
                    risk_factors.append("⚠️ Low peak flow readings may indicate restricted airways")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.markdown("✅ No significant risk factors identified")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("If you have access to the preprocessing pipeline, you can check the exact age normalization formula used during model training.")


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
                # st.balloons()
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


if selected == 'Medical Consultant':
    st.title("🩺 Medical Consultant Chatbot")
    st.markdown("### Discuss Your Health Concerns with Our AI-powered Chatbot")
    st.write("Ask about **Diabetes, Asthma, Stroke, Cardiovascular Disease, or Mental Health.**")

    genai.configure(api_key="AIzaSyAwyi9c5OdvLoWrv5lFi1jZDEYwuprQAKE")

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
            model = genai.GenerativeModel("gemini-2.0-flash")
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


# if selected == 'Checkbox-to-disease-predictor':
# # Load transformer model
#     classifier = pipeline("zero-shot-classification", model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
#     # Use a pipeline as a high-level helper

#     # from transformers import pipeline

#     # pipe = pipeline("fill-mask", model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")


#     # Define symptoms for each disease
#     diseases = {
#         "Diabetes": ["Frequent urination", "Increased thirst", "Unexplained weight loss", "Fatigue", "Blurred vision"],
#         "Hypertension": ["Headache", "Dizziness", "Chest pain", "Shortness of breath", "Nosebleeds"],
#         "Obesity": ["Excess body fat", "Breathlessness", "Joint pain", "Increased sweating", "Low energy levels"],
#         "Cardiovascular Disease": ["Chest pain", "Shortness of breath", "Dizziness", "Irregular heartbeat", "Fatigue"],
#         "COPD": ["Chronic cough", "Shortness of breath", "Wheezing", "Chest tightness", "Frequent respiratory infections"],
#         "Liver Disease": ["Jaundice", "Abdominal pain", "Swelling in legs", "Chronic fatigue", "Nausea"],
#         "Kidney Disease": ["Swelling in legs", "Fatigue", "Loss of appetite", "Changes in urination", "Muscle cramps"],
#         "Metabolic Syndrome": ["High blood sugar", "High blood pressure", "Increased waist size", "High cholesterol", "Fatigue"],
#         "Osteoarthritis": ["Joint pain", "Stiffness", "Swelling", "Reduced flexibility", "Bone spurs"],
#         "Gastroesophageal Reflux Disease": ["Heartburn", "Acid reflux", "Difficulty swallowing", "Chronic cough", "Sore throat"],
#         "Depression": ["Persistent sadness", "Loss of interest", "Sleep disturbances", "Fatigue", "Difficulty concentrating"],
#         "Sleep Apnea": ["Loud snoring", "Pauses in breathing", "Daytime drowsiness", "Morning headaches", "Irritability"],
#     }

#     # Streamlit UI
#     st.title("🩺 Hybrid Symptom Checker")
#     st.write("Select your symptoms and get AI-powered predictions!")

#     selected_symptoms = []

#     # Create symptom selection with markdown separation and three columns
#     disease_keys = list(diseases.keys())

#     for i in range(0, len(disease_keys), 3):
#         cols = st.columns(3)
#         for j in range(3):
#             if i + j < len(disease_keys):
#                 disease = disease_keys[i + j]
#                 with cols[j]:
#                     st.markdown(f"### {disease}")
#                     for symptom in diseases[disease]:
#                         if st.checkbox(symptom, key=f"{disease}_{symptom}"):
#                             selected_symptoms.append(symptom)

#     if st.button("🔍 Predict Disease"):
#         if selected_symptoms:
#             user_input = ", ".join(selected_symptoms)  # Convert symptoms to text

#             # 1️⃣ Custom Symptom Matching Approach
#             disease_scores = {disease: 0 for disease in diseases.keys()}
#             for disease, symptoms in diseases.items():
#                 matches = sum(symptom in selected_symptoms for symptom in symptoms)
#                 disease_scores[disease] = matches / len(symptoms)  # Normalize by symptom count

#             # Normalize to percentage
#             symptom_match_scores = {d: round(score * 100, 2) for d, score in disease_scores.items()}

#             # 2️⃣ AI Model Prediction
#             ai_results = classifier(user_input, list(diseases.keys()))
#             ai_scores = {ai_results["labels"][i]: round(ai_results["scores"][i] * 100, 2) for i in range(len(ai_results["labels"]))}

#             # 3️⃣ Hybrid Score Calculation (Average of Both Scores)
#             final_scores = {}
#             for disease in diseases.keys():
#                 symptom_score = symptom_match_scores.get(disease, 0)
#                 ai_score = ai_scores.get(disease, 0)
#                 final_scores[disease] = round((symptom_score + ai_score) / 2, 2)  # Averaging

#             # Sort by final score
#             sorted_final_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

#             # Display results
#             st.write("### 🔬 Possible Conditions (Hybrid Model Prediction):")
#             for disease, score in sorted_final_scores:
#                 if score > 0:
#                     st.write(f"🩺 {disease}: {score}% match")
#         else:
#             st.write("⚠️ Please select at least one symptom.")


import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer



if selected == "Mental-Analysis":
    # Load the Hugging Face model
    model_name = "mental/mental-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Sidebar with title and markdown
    st.sidebar.title("🧠 Mental Health Analysis")
    st.sidebar.markdown("""
    Analyze mental health symptoms using a **pre-trained AI model**.  
    This tool predicts **Depression and Anxiety** based on text input.
    """)

    # Main content
    st.title("🔬 Mental Health Text Analysis")
    st.markdown("Enter a description of your mental state, and the AI will predict possible conditions.")

    # User input
    user_input = st.text_area("Describe your symptoms (e.g., 'I feel hopeless and anxious all the time.'):")
    
    if st.button("Analyze"):
        if user_input:
            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

            # Get raw logits from the model
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits

            # Apply sigmoid activation to get independent probabilities
            probs = torch.sigmoid(logits).squeeze().tolist()

            # Map to labels
            label_mapping = {
                0: "Depression",
                1: "Anxiety"
            }
            predictions = {label_mapping[i]: round(probs[i] * 100, 2) for i in range(len(probs))}

            # Display predictions
            st.write("### Predictions:")
            for label, score in predictions.items():
                st.write(f"🩺 **{label}**: {score}% confidence")

            # Sort for better visualization
            sorted_labels = sorted(predictions.keys(), key=lambda x: predictions[x], reverse=True)
            sorted_scores = [predictions[label] for label in sorted_labels]

            # Plot using Seaborn
            fig, ax = plt.subplots(figsize=(4, 2.5))  # Compact size
            sns.barplot(x=sorted_scores, y=sorted_labels, palette="coolwarm", ax=ax)

            # Labels & title
            ax.set_xlabel("Risk Probability (%)")
            ax.set_title("Mental Health Risk Assessment")
            ax.set_xlim(0, 100)
            
            # Add percentages inside bars
            for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
                ax.text(score - 5, i, f"{score}%", va='center', ha='right', color='white', fontsize=10, fontweight='bold')

            # Display the chart in a single column
            st.pyplot(fig)


if selected == 'Sleep Health Analysis':
    st.title("🌙 Sleep Health Analysis")
    st.image("https://cdn-icons-png.flaticon.com/512/1205/1205526.png", width=100)
    
    st.markdown("""
    This model predicts the likelihood of **Sleep Disorders** based on various health factors.  
    Enter your details and click **"Sleep Health Test Result"** to get the prediction.
    """)

    # Load models
    try:
        sleep_model = pickle.load(open('sleep_health/svc_model.pkl', 'rb'))
        scaler = pickle.load(open('sleep_health/scaler.pkl', 'rb'))
        label_encoder = pickle.load(open('sleep_health/label_encoders.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Error: Model files not found. Please upload the model files.")
        st.stop()

    # Input fields for user data
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'], key='gender_sleep')
        age = st.slider("Age", min_value=27, max_value=59, value=35, 
                      help="Age range in dataset: 27-59 years", key='age_sleep')
        occupation = st.selectbox("Occupation", 
            ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Business',
             'Scientist', 'Accountant', 'Engineer'], key='occupation_sleep')
        sleep_duration = st.slider("Sleep Duration (hours)", 
            min_value=5.8, max_value=8.5, value=6.5, step=0.1,
            help="Dataset range: 5.8-8.5 hours", key='sleep_duration')
        quality_of_sleep = st.slider('Quality of Sleep', 
            min_value=4, max_value=9, value=6,
            help="Higher is better (4-9 scale)", key='quality_sleep')
        physical_activity_level = st.slider('Physical Activity Level (minutes/day)', 
            min_value=30, max_value=90, value=45,
            help="Physical activity in minutes per day", key='activity_sleep')

    with col2:
        stress_level = st.slider('Stress Level', 
            min_value=3, max_value=8, value=6, 
            help="Higher values indicate higher stress (3-8 scale)", key='stress_sleep')
        bmi_category = st.selectbox("BMI Category", 
            ["Normal", "Overweight", "Obese"], key='bmi_sleep')
        
        # For blood pressure, let's use two separate inputs for systolic/diastolic
        col2a, col2b = st.columns(2)
        with col2a:
            systolic = st.slider("Blood Pressure (Systolic)", 
                min_value=110, max_value=140, value=125, key='bp_sys_sleep')
        with col2b:
            diastolic = st.slider("Diastolic", 
                min_value=70, max_value=95, value=80, key='bp_dia_sleep')
        
        blood_pressure = f"{systolic}/{diastolic}"
        
        heart_rate = st.slider("Heart Rate (bpm)", 
            min_value=65, max_value=86, value=75, 
            help="Normal range: 60-100 bpm", key='hr_sleep')
        daily_steps = st.slider("Daily Steps", 
            min_value=3000, max_value=10000, value=6000, step=500,
            help="Recommended: 7,000-10,000 steps/day", key='steps_sleep')

    # Create a button to trigger prediction
    if st.button('Sleep Health Test Result', key='sleep_test_button'):
        try:
            # Prepare input data
            input_data = {
                'Gender': gender,
                'Age': age,
                'Occupation': occupation,
                'Sleep Duration': sleep_duration,
                'Quality of Sleep': quality_of_sleep,
                'Physical Activity Level': physical_activity_level,
                'Stress Level': stress_level,
                'BMI Category': bmi_category,
                'Blood Pressure': blood_pressure,
                'Heart Rate': heart_rate,
                'Daily Steps': daily_steps
            }
            
            # Process and predict
            df = pd.DataFrame([input_data])
            
            # Apply label encoding to categorical features
            for col, encoder in label_encoder.items():
                if col in df.columns:
                    df[col] = encoder.transform([df[col].iloc[0]])[0]
            
            # Feature Engineering and Preprocessing
            columns_to_drop = ["Physical Activity Level", "Person ID"]
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Create dummy variables with the correct column names expected by the model
            df = pd.get_dummies(df)
            
            # Get the expected feature names from the model
            # You might need to store these feature names during training
            expected_features = [
                'Age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level',
                'Heart Rate', 'Daily Steps', 'Gender_Female', 'Gender_Male',
                'Occupation_Accountant', 'Occupation_Business', 'Occupation_Doctor',
                'Occupation_Engineer', 'Occupation_Sales Representative', 
                'Occupation_Scientist', 'Occupation_Software Engineer', 'Occupation_Teacher',
                'BMI Category_Normal', 'BMI Category_Obese', 'BMI Category_Overweight',
                'Blood Pressure_110/70', 'Blood Pressure_120/80', 'Blood Pressure_125/80',
                'Blood Pressure_130/85', 'Blood Pressure_140/90'
            ]
            
            # Create a DataFrame with expected features filled with zeros
            prediction_df = pd.DataFrame(0, index=[0], columns=expected_features)
            
            # Fill in the values from our current DataFrame
            for col in df.columns:
                if col in prediction_df.columns:
                    prediction_df[col] = df[col].values
                    
            # Scale with proper error handling
            with st.spinner("⏳ Predicting... Please wait..."):
                time.sleep(2)
                # Use the properly formatted DataFrame
                prediction = sleep_model.predict(prediction_df)
            
            # Display result
            result = "🛑 High risk of sleep disorder" if prediction[0] == 1 else "✅ Low risk of sleep disorder"
            if prediction[0] == 0:
                st.balloons()
            st.success(result)
            
            # Show risk factors based on input
            st.subheader("Risk Factor Analysis")
            risk_factors = []
            
            if sleep_duration < 6.0:
                risk_factors.append("⚠️ Low sleep duration (less than 6 hours)")
            if quality_of_sleep < 6:
                risk_factors.append("⚠️ Poor sleep quality")
            if stress_level > 6:
                risk_factors.append("⚠️ High stress levels")
            if bmi_category in ["Overweight", "Obese"]:
                risk_factors.append(f"⚠️ {bmi_category} BMI category")
            if int(systolic) > 130 or int(diastolic) > 85:
                risk_factors.append("⚠️ Elevated blood pressure")
            if heart_rate > 80:
                risk_factors.append("⚠️ Elevated heart rate")
            if daily_steps < 5000:
                risk_factors.append("⚠️ Low daily activity (steps)")
                
            if risk_factors:
                st.markdown("##### Potential Risk Factors:")
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.markdown("✅ No significant risk factors identified.")

        except Exception as e:
            st.error(f"❌ Error: {e}")




if selected=='Hypertension Prediction':
    st.title("Hypertension Risk Prediction App")
    st.markdown("This application uses an Extra Trees Classifier model to predict hypertension risk based on patient health data.")

    # Load the model and scaler
    try:
        hypertension_model = pickle.load(open('hypertension\extratrees_model.pkl', 'rb'))
        hypertension_scaler = pickle.load(open('hypertension\scaler.pkl', 'rb'))
        st.success("Model and scaler loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.info("Please check that model and scaler files are in the correct location.")
        st.warning("Expected path: 'hypertension/extratrees_model.pkl' and 'hypertension/scaler.pkl'")

    # Define input section
    st.subheader("Patient Information")

    # Create two columns for input layout
    col1, col2 = st.columns(2)

    with col1:
        male = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.slider("Age", min_value=32, max_value=70, value=49, help="Patient's age (32-70 years)")
        cigs_per_day = st.slider("Cigarettes Per Day", min_value=0.0, max_value=70.0, value=0.0, step=1.0)
        bp_meds = st.radio("On Blood Pressure Medication", options=[0.0, 1.0], format_func=lambda x: "No" if x == 0.0 else "Yes")
        tot_chol = st.slider("Total Cholesterol", min_value=107.0, max_value=500.0, value=234.0, step=1.0, help="mg/dL")

    with col2:
        sys_bp = st.slider("Systolic Blood Pressure", min_value=83.5, max_value=295.0, value=128.0, step=0.5, help="mmHg")
        dia_bp = st.slider("Diastolic Blood Pressure", min_value=48.0, max_value=142.5, value=82.0, step=0.5, help="mmHg")
        bmi = st.slider("BMI", min_value=15.54, max_value=56.80, value=25.40, step=0.01)
        heart_rate = st.slider("Heart Rate", min_value=44.0, max_value=143.0, value=75.0, step=1.0, help="beats per minute")
        glucose = st.slider("Glucose", min_value=40.0, max_value=394.0, value=78.0, step=1.0, help="mg/dL")

    # Prediction button
    predict_button = st.button("Predict Hypertension Risk")

    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame({
            'male': [male],
            'age': [age],
            'cigsPerDay': [cigs_per_day],
            'BPMeds': [bp_meds],
            'totChol': [tot_chol],
            'sysBP': [sys_bp],
            'diaBP': [dia_bp],
            'BMI': [bmi],
            'heartRate': [heart_rate],
            'glucose': [glucose]
        })
        
        # Display input data
        st.subheader("Input Data:")
        st.dataframe(input_data)
        
        # Identify numerical columns to scale
        num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        
        try:
            # Scale the numerical features
            input_data[num_cols] = hypertension_scaler.transform(input_data[num_cols])
            
            # Make prediction
            prediction = hypertension_model.predict(input_data)[0]
            prediction_prob = hypertension_model.predict_proba(input_data)[0]
            
            # Display prediction results
            st.subheader("Prediction Result:")
            
            # Create columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 0:
                    st.success("✅ Low Risk of Hypertension")
                else:
                    st.error("🚨 High Risk of Hypertension")
            
            with res_col2:
                # Visualization
                st.write(f"Probability of Low Risk: {prediction_prob[0]:.2f}")
                st.write(f"Probability of High Risk: {prediction_prob[1]:.2f}")
                
                # Add progress bar
                st.progress(float(prediction_prob[1]))
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
        # st.info("Please check that all inputs are valid and within the expected ranges.")