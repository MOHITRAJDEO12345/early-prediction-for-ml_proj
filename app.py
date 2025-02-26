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
        options=['Home', 'Diabetes Prediction', 'Asthma Prediction', 'Cardiovascular Disease Prediction', 'Stroke Prediction', 'Data Visualization', 'Chat with us', 'Text-to-disease-predictor', 'Checkbox-to-disease-predictor'],
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
    st.title("🩺 Multiple Disease Prediction System")

    st.markdown("""
    ## Welcome to the **AI-Powered Health Prediction System**!  
    This tool helps predict the likelihood of **four major diseases** using **Machine Learning**:
    
    - **🩸 Diabetes**
    - **🌬️ Asthma**
    - **🧠 Stroke**
    - **❤️ Cardiovascular Disease**
    
    👉 Select a prediction model from the sidebar to proceed!  
    """)

    with st.expander("🚀 Quick Start Guide"):
        st.write("""
        1. Select a disease prediction model from the sidebar.
        2. Enter your medical details in the input fields.
        3. Click the **Predict** button to get your result.
        4. Read health recommendations based on your result.
        """)

    st.markdown("""
        ### 📚 Learn More About These Diseases
        - [What is Diabetes?](https://www.example.com)
        - [Understanding Asthma](https://www.example.com)
        - [Stroke Prevention & Recovery](https://www.example.com)
        - [Heart & Cardiovascular Health](https://www.example.com)
        """)

    # ⭐ User Rating System
    rating = st.slider("⭐ Rate this app", 1, 5)
    if st.button('Submit Rating'):
        st.success(f"✅ Thank you for rating us {rating} stars!")

    # 💬 User Feedback Section
    feedback = st.text_area("💬 Provide Feedback", placeholder="How was your experience?")
    if st.button('Submit Feedback'):
        st.success("✅ Thank you for your feedback!")

    # 📩 Contact Information
    st.markdown("""
        ### 📩 Contact Us
        Have questions? Email us at [mohitrajdeo16deoghar@gmail.com](mailto:mohitrajdeo16deoghar@gmail.com).
        """)

    # Disclaimer Section
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:** This app is for educational purposes only and should not replace professional medical advice.
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


if selected == 'Chat with us':
    st.title("👩‍💻 Chat with us")
    st.markdown("### Let's chat about your health concerns!")
    st.write("Ask about **Diabetes, Asthma, Stroke, or Cardiovascular Disease**.")

    # load_dotenv()

    # Get the Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ API Key not found! Please check your .env file.")
        st.stop()

    # Configure Gemini API
    genai.configure(api_key="AIzaSyD9x7Kz8adDo6-nVyk9MAQjlwD4lTeKc84")

    # Custom Styling
    st.markdown("""
        <style>
            .prompt-box { 
                background-color: #000000; 
                padding: 10px; 
                border-radius: 8px; 
                font-size: 14px; 
                font-family: monospace;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🩺 Health Specialist Chatbot")
    st.markdown("### Get expert advice on **Diabetes, Asthma, Stroke, and Cardiovascular Disease!**")
    st.write("Ask me about symptoms, diet, exercise, and lifestyle recommendations.")

    # Predefined Prompts for Four Diseases
    st.markdown("#### 💡 Quick Prompts (Click to Copy)")
    
    prompt_options = {
        "Diabetes – Diet": "What foods should I eat if I have diabetes?",
        "Diabetes – Exercise": "What type of workouts help control blood sugar levels?",
        "Asthma – Triggers": "What are common asthma triggers?",
        "Asthma – Treatment": "What are the best medications for asthma?",
        "Stroke – Symptoms": "What are the early warning signs of a stroke?",
        "Stroke – Recovery": "How long does it take to recover from a stroke?",
        "Cardiovascular – Heart Health": "How can I reduce my risk of heart disease?",
        "Cardiovascular – Blood Pressure": "What lifestyle changes can lower high blood pressure?"
    }

    for label, prompt in prompt_options.items():
        st.markdown(f"""<div class="prompt-box"><strong>{label}</strong><br>{prompt}</div>""", unsafe_allow_html=True)
        st.code(prompt, language="text")

    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input field
    user_prompt = st.chat_input("Ask about Diabetes, Asthma, Stroke, or Cardiovascular Disease...")

    if user_prompt:
        # Display user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Gemini API request
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


if selected == 'Text-to-disease-predictor':
    st.title("🔮 Text-to-Disease Predictor")
    st.markdown("### Enter your symptoms to predict the likelihood of a disease!")
    st.write("Try entering symptoms like 'I have a fever and cough'.")
    st.write("This tool uses a pre-trained model to predict the likelihood of common diseases based on your symptoms.")

    # Load the pre-trained model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Streamlit UI
    st.title("Text-Based Symptom Analysis")
    user_input = st.text_area("Enter your symptoms (e.g., 'I have a fever and cough'):")

    # Define candidate diseases
    candidate_labels = ["influenza", "COVID-19", "common cold", "pneumonia", "bronchitis"]

    if st.button("Predict Disease"):
        if user_input:
            result = classifier(user_input, candidate_labels)
            st.write("Possible Conditions:")
            for disease, score in zip(result["labels"], result["scores"]):
                st.write(f"🩺 {disease}: {round(score * 100, 2)}% risk")



if selected == 'Checkbox-to-disease-predictor':
    st.title("🔮 Checkbox-to-Disease Predictor")
    st.markdown("### Select symptoms to predict the likelihood of a disease!")
    st.write("This tool uses a pre-trained model to predict the likelihood of common diseases based on your selected symptoms.")

