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
        options=['Home','Text-based Disease Prediction', 'Checkbox-to-disease-predictor', 'AI Health Consultant', 'Mental-Analysis', 'Diabetes Prediction', 'Asthma Prediction', 'Cardiovascular Disease Prediction', 'Stroke Prediction', 'Sleep Health Analysis', 'Data Visualization'],
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
    - [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)  
    - [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)  
    - [Sleep Health Analysis](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
                
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
    st.markdown("Analyze sleep patterns and health conditions using a **pre-trained AI model**.")
    st.markdown("This tool predicts **Sleep Disorders** based on input data.")

    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Load the trained model
    try:
        with open('sleep_health/best_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: 'best_model.pkl' not found. Please upload the model file.")
        st.stop()

    # Load the scaler
    try:
        with open('sleep_health/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: 'scaler.pkl' not found. Please upload the scaler file.")
        st.stop()

    # Input fields for user data
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input("Age", min_value=18, max_value=100)
    occupation = st.selectbox("Occupation", ['Software Engineer', 'Teacher', 'Doctor', 'Business', 'Sales Representative', 'Scientist', 'Accountant', 'Engineer'])
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0)
    quality_of_sleep = st.number_input('Quality of Sleep', min_value=1, max_value=5)
    physical_activity_level = st.number_input('Physical Activity Level', min_value=1, max_value=5)
    stress_level = st.number_input('Stress Level', min_value=1, max_value=5)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
    heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000)

    # Create a button to trigger prediction
    # if st.button("Predict"):
    #     # Prepare input data
    #     input_data = pd.DataFrame({
    #         'Gender': [gender],
    #         'Age': [age],
    #         'Occupation': [occupation],
    #         'Sleep Duration': [sleep_duration],
    #         'Quality of Sleep': [quality_of_sleep],
    #         'Physical Activity Level': [physical_activity_level],
    #         'Stress Level': [stress_level],
    #         'BMI Category': [bmi_category],
    #         'Blood Pressure': [blood_pressure],
    #         'Heart Rate': [heart_rate],
    #         'Daily Steps': [daily_steps]
    #     })

    #     # Preprocess the input data
    #     try:
    #         # Encode categorical features
    #         input_data['Gender'] = LabelEncoder().fit_transform(input_data['Gender'])
    #         input_data['Occupation'] = LabelEncoder().fit_transform(input_data['Occupation'])
    #         input_data = pd.get_dummies(input_data, drop_first=True)

    #         X_train = 0
    #         with open('sleep_health/columns.pkl', 'wb') as file:
    #             pickle.dump(columns, file)            # Handle missing columns in the test set
    #         missing_cols = set(X_train.columns) - set(input_data.columns)
    #         for c in missing_cols:
    #             input_data[c] = 0
    #         input_data = input_data[X_train.columns]

    #         # Scale the input data
    #         input_data = scaler.transform(input_data)

    #         # Make a prediction using the loaded model
    #         prediction = model.predict(input_data)
    #         predicted_class = le.inverse_transform(prediction)[0]  # Convert the prediction back to original label
    #         st.write(f"Predicted Sleep Disorder: {predicted_class}")
    #     except ValueError as e:  # Catch any errors that could occur with preprocessing
    #         st.error(f"Error during prediction: {e}")
    



# if selected == 'Text-based Disease Prediction':
#     st.title("📝 Text-based Disease Prediction")
#     st.markdown("Enter your symptoms in the text area below, or use audio input, and the AI will predict possible lifestyle diseases.")

#     from transformers import AutoTokenizer, AutoModelForSequenceClassification
#     import torch
#     import speech_recognition as sr
#     from pydub import AudioSegment
#     from pydub.playback import play

#     # Define diseases and symptoms
#     diseases = {
#         "Diabetes": ["Frequent urination", "Increased thirst", "Unexplained weight loss", "Fatigue", "Blurred vision", "Slow-healing wounds", "Increased hunger", "Dry skin", "Numbness or tingling in hands/feet", "Recurring infections"],
#         "Hypertension": ["Headache", "Dizziness", "Chest pain", "Shortness of breath", "Nosebleeds", "Flushing", "Vision problems", "Irregular heartbeat", "Blood in urine", "Fatigue"],
#         "Obesity": ["Excess body fat", "Breathlessness", "Joint pain", "Increased sweating", "Low energy levels", "Sleep apnea", "Skin problems", "Back pain", "Difficulty with physical activity", "High blood pressure"],
#         "Cardiovascular Disease": ["Chest pain", "Shortness of breath", "Dizziness", "Irregular heartbeat", "Fatigue", "Swelling in legs/ankles", "Neck/jaw/throat/back pain", "Nausea", "Cold sweats", "Lightheadedness"],
#         "COPD": ["Chronic cough", "Shortness of breath", "Wheezing", "Chest tightness", "Frequent respiratory infections", "Bluish lips or fingernails", "Fatigue", "Unintended weight loss", "Swelling in ankles/feet/legs", "Difficulty sleeping"],
#         "Liver Disease": ["Jaundice", "Abdominal pain", "Swelling in legs", "Chronic fatigue", "Nausea", "Itchy skin", "Dark urine", "Pale stools", "Loss of appetite", "Easy bruising"],
#         "Kidney Disease": ["Swelling in legs", "Fatigue", "Loss of appetite", "Changes in urination", "Muscle cramps", "Nausea", "Difficulty concentrating", "Dry, itchy skin", "Shortness of breath", "High blood pressure"],
#         "Metabolic Syndrome": ["High blood sugar", "High blood pressure", "Increased waist size", "High cholesterol", "Fatigue", "Blurred vision", "Increased thirst", "Frequent urination", "Slow-healing wounds", "Skin tags"],
#         "Osteoarthritis": ["Joint pain", "Stiffness", "Swelling", "Reduced flexibility", "Bone spurs", "Grating sensation", "Tenderness", "Loss of joint space", "Muscle weakness", "Deformity"],
#         "Gastroesophageal Reflux Disease": ["Heartburn", "Acid reflux", "Difficulty swallowing", "Chronic cough", "Sore throat", "Chest pain", "Regurgitation", "Feeling of lump in throat", "Hoarseness", "Bad breath"],
#         "Depression": ["Persistent sadness", "Loss of interest", "Sleep disturbances", "Fatigue", "Difficulty concentrating", "Changes in appetite", "Feelings of worthlessness", "Irritability", "Physical aches and pains", "Thoughts of death or suicide"],
#         "Sleep Apnea": ["Loud snoring", "Pauses in breathing", "Daytime drowsiness", "Morning headaches", "Irritability", "Dry mouth upon waking", "Difficulty staying asleep", "Attention problems", "Mood changes", "High blood pressure"],
#         "Asthma": ["Wheezing", "Shortness of breath", "Chest tightness", "Coughing", "Difficulty sleeping", "Fatigue", "Anxiety", "Rapid breathing", "Difficulty speaking", "Blue lips or fingernails"],
#         "Rheumatoid Arthritis": ["Joint pain", "Swelling", "Stiffness", "Fatigue", "Fever", "Loss of appetite", "Dry eyes and mouth", "Firm bumps under the skin", "Numbness and tingling", "Anemia"],
#         "Alzheimer's Disease": ["Memory loss", "Difficulty planning or solving problems", "Trouble completing familiar tasks", "Confusion with time or place", "Vision problems", "Problems with words", "Misplacing things", "Poor judgment", "Withdrawal from social activities", "Mood and personality changes"]
#     }

#     # Load the pre-trained model and tokenizer
#     model_name = "distilbert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(diseases))

#     # Create a text area for user input
#     user_input = st.text_area("Describe your symptoms (e.g., 'I feel tired all the time and have frequent headaches.'):")

#     # Audio input
#     st.markdown("### Or use audio input:")
#     audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

#     if audio_file is not None:
#         # Convert audio file to text
#         recognizer = sr.Recognizer()
#         audio = AudioSegment.from_file(audio_file)
#         audio.export("temp.wav", format="wav")
#         with sr.AudioFile("temp.wav") as source:
#             audio_data = recognizer.record(source)
#             try:
#                 user_input = recognizer.recognize_google(audio_data)
#                 st.write(f"Recognized Text: {user_input}")
#             except sr.UnknownValueError:
#                 st.write("Google Speech Recognition could not understand the audio.")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results from Google Speech Recognition service; {e}")

#     if st.button("Predict Disease"):
#         if user_input:
#             # Tokenize and preprocess the input text
#             inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

#             # Get raw logits from the model
#             with torch.no_grad():
#                 outputs = model(**inputs)
#             logits = outputs.logits

#             # Apply softmax to get probabilities
#             probs = torch.softmax(logits, dim=1).squeeze().tolist()

#             # Map to labels
#             disease_labels = list(diseases.keys())
#             predictions = {disease_labels[i]: round(probs[i] * 100, 2) for i in range(len(probs))}

#             # Display predictions
#             st.write("### Predictions:")
#             for label, score in predictions.items():
#                 st.write(f"🩺 **{label}**: {score}% confidence")

#             # Sort for better visualization
#             sorted_labels = sorted(predictions.keys(), key=lambda x: predictions[x], reverse=True)
#             sorted_scores = [predictions[label] for label in sorted_labels]

#             # Plot using Seaborn
#             fig, ax = plt.subplots(figsize=(4, 2.5))  # Compact size
#             sns.barplot(x=sorted_scores, y=sorted_labels, palette="coolwarm", ax=ax)

#             # Labels & title
#             ax.set_xlabel("Risk Probability (%)")
#             ax.set_title("Disease Risk Assessment")
#             ax.set_xlim(0, 100)
            
#             # Add percentages inside bars
#             for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
#                 ax.text(score - 5, i, f"{score}%", va='center', ha='right', color='white', fontsize=10, fontweight='bold')

#             # Display the chart in a single column
#             st.pyplot(fig)
#         else:
#             st.write("⚠️ Please enter your symptoms.")




# if selected == 'Text-based Disease Prediction':
#     st.title("📝 Text-based Disease Prediction")
#     st.markdown("Enter your symptoms in the text area below, or use audio input, and the AI will predict possible lifestyle diseases.")

#     import os
#     import google.generativeai as genai
#     import speech_recognition as sr
#     from pydub import AudioSegment
#     from pydub.playback import play

#     # Configure the Gemini API
#     genai.configure(api_key="AIzaSyD9x7Kz8adDo6-nVyk9MAQjlwD4lTeKc84")

#     # Define diseases and symptoms
#     diseases = {
#         "Diabetes": ["Frequent urination", "Increased thirst", "Unexplained weight loss", "Fatigue", "Blurred vision", "Slow-healing wounds", "Increased hunger", "Dry skin", "Numbness or tingling in hands/feet", "Recurring infections"],
#         "Hypertension": ["Headache", "Dizziness", "Chest pain", "Shortness of breath", "Nosebleeds", "Flushing", "Vision problems", "Irregular heartbeat", "Blood in urine", "Fatigue"],
#         "Obesity": ["Excess body fat", "Breathlessness", "Joint pain", "Increased sweating", "Low energy levels", "Sleep apnea", "Skin problems", "Back pain", "Difficulty with physical activity", "High blood pressure"],
#         "Cardiovascular Disease": ["Chest pain", "Shortness of breath", "Dizziness", "Irregular heartbeat", "Fatigue", "Swelling in legs/ankles", "Neck/jaw/throat/back pain", "Nausea", "Cold sweats", "Lightheadedness"],
#         "COPD": ["Chronic cough", "Shortness of breath", "Wheezing", "Chest tightness", "Frequent respiratory infections", "Bluish lips or fingernails", "Fatigue", "Unintended weight loss", "Swelling in ankles/feet/legs", "Difficulty sleeping"],
#         "Liver Disease": ["Jaundice", "Abdominal pain", "Swelling in legs", "Chronic fatigue", "Nausea", "Itchy skin", "Dark urine", "Pale stools", "Loss of appetite", "Easy bruising"],
#         "Kidney Disease": ["Swelling in legs", "Fatigue", "Loss of appetite", "Changes in urination", "Muscle cramps", "Nausea", "Difficulty concentrating", "Dry, itchy skin", "Shortness of breath", "High blood pressure"],
#         "Metabolic Syndrome": ["High blood sugar", "High blood pressure", "Increased waist size", "High cholesterol", "Fatigue", "Blurred vision", "Increased thirst", "Frequent urination", "Slow-healing wounds", "Skin tags"],
#         "Osteoarthritis": ["Joint pain", "Stiffness", "Swelling", "Reduced flexibility", "Bone spurs", "Grating sensation", "Tenderness", "Loss of joint space", "Muscle weakness", "Deformity"],
#         "Gastroesophageal Reflux Disease": ["Heartburn", "Acid reflux", "Difficulty swallowing", "Chronic cough", "Sore throat", "Chest pain", "Regurgitation", "Feeling of lump in throat", "Hoarseness", "Bad breath"],
#         "Depression": ["Persistent sadness", "Loss of interest", "Sleep disturbances", "Fatigue", "Difficulty concentrating", "Changes in appetite", "Feelings of worthlessness", "Irritability", "Physical aches and pains", "Thoughts of death or suicide"],
#         "Sleep Apnea": ["Loud snoring", "Pauses in breathing", "Daytime drowsiness", "Morning headaches", "Irritability", "Dry mouth upon waking", "Difficulty staying asleep", "Attention problems", "Mood changes", "High blood pressure"],
#         "Asthma": ["Wheezing", "Shortness of breath", "Chest tightness", "Coughing", "Difficulty sleeping", "Fatigue", "Anxiety", "Rapid breathing", "Difficulty speaking", "Blue lips or fingernails"],
#         "Rheumatoid Arthritis": ["Joint pain", "Swelling", "Stiffness", "Fatigue", "Fever", "Loss of appetite", "Dry eyes and mouth", "Firm bumps under the skin", "Numbness and tingling", "Anemia"],
#         "Alzheimer's Disease": ["Memory loss", "Difficulty planning or solving problems", "Trouble completing familiar tasks", "Confusion with time or place", "Vision problems", "Problems with words", "Misplacing things", "Poor judgment", "Withdrawal from social activities", "Mood and personality changes"]
#     }

#     # Create a text area for user input
#     user_input = st.text_area("Describe your symptoms (e.g., 'I feel tired all the time and have frequent headaches.'):")

#     # Audio input
#     st.markdown("### Or use audio input:")
#     audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

#     if audio_file is not None:
#         # Convert audio file to text
#         recognizer = sr.Recognizer()
#         audio = AudioSegment.from_file(audio_file)
#         audio.export("temp.wav", format="wav")
#         with sr.AudioFile("temp.wav") as source:
#             audio_data = recognizer.record(source)
#             try:
#                 user_input = recognizer.recognize_google(audio_data)
#                 st.write(f"Recognized Text: {user_input}")
#             except sr.UnknownValueError:
#                 st.write("Google Speech Recognition could not understand the audio.")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results from Google Speech Recognition service; {e}")

#     if st.button("Predict Disease"):
#         if user_input:
#             # Use Gemini model to generate predictions
#             model = genai.GenerativeModel("gemini-2.0-flash-lite")
#             response = model.generate_content(user_input)

#             if response and hasattr(response, "text"):
#                 predictions = response.text
#             else:
#                 predictions = "I'm sorry, I couldn't generate a response."

#             # Display predictions
#             st.write("### Predictions:")
#             st.write(predictions)

#             # Sort for better visualization
#             sorted_labels = sorted(predictions.keys(), key=lambda x: predictions[x], reverse=True)
#             sorted_scores = [predictions[label] for label in sorted_labels]

#             # Plot using Seaborn
#             fig, ax = plt.subplots(figsize=(4, 2.5))  # Compact size
#             sns.barplot(x=sorted_scores, y=sorted_labels, palette="coolwarm", ax=ax)

#             # Labels & title
#             ax.set_xlabel("Risk Probability (%)")
#             ax.set_title("Disease Risk Assessment")
#             ax.set_xlim(0, 100)
            
#             # Add percentages inside bars
#             for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
#                 ax.text(score - 5, i, f"{score}%", va='center', ha='right', color='white', fontsize=10, fontweight='bold')

#             # Display the chart in a single column
#             st.pyplot(fig)
#         else:
#             st.write("⚠️ Please enter your symptoms.")