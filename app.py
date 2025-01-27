#!/usr/bin/env python
# coding: utf-8

# ## **UI - User Interface for Model Prediction**

# In[6]:
#build a streamlit app (app.py)
import streamlit as st
st.set_page_config(page_title="FraudInspectAI", page_icon="\U0001F50D", layout="centered")



import numpy as np
import joblib
import matplotlib.pyplot as plt


# In[7]:
# Add CSS for dark app background, white text, and orange sidebar background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #111111; /* Dark background for the app */
        color: #ffffff; /* White text for all elements */
    }
    .title {
        font-size: 36px;
        color: #ff6b00; /* Orange for the title */
        font-weight: bold;
    }
    .subheader {
        font-size: 24px;
        color: #ff9800; /* Orange for subheaders */
    }
    .stSidebar {
        background-color: #000000; /* black background for the sidebar */
    }
    .stSidebar > div {
        color: #ff6b00; /* Ensure sidebar text is orange */
    }
    .stNumberInput > label, .stButton > button {
        color: #808080 !important; /* gray labels and buttons */
    }
    .stButton > button:hover {
        background-color: #808080; /* Button hover effect */
        color: #808080; /* Keep button text gray */
    }
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif; /* Clean and modern font */
    }
    </style>
    """,
    unsafe_allow_html=True

)


# load the trained model (combined dataset model)
model = joblib.load("credit_card_combined_model.pkl")



# In[8]:


#define the UI | page config and logo
st.image("fraudinspectai_logo.jpg", use_container_width=True)  #logo width adjustable

st.markdown('<div class="title">FraudInspectAI - Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.write("Enter transaction details to predict if it's fraudulent.")


# In[10]:


#Sidebar for user input
st.sidebar.header("Enter Transaction Details")
input_values = []
for i in range(1, 29):
    input_values.append(st.number_input(f"V{i}", value=0.0))

amount = st.number_input("Transaction Amount", value=0.0)
input_values.append(amount)


# convert input into a numpy array
input_array = np.array([input_values])

#Initialize result and probability
result = ""
probability = 0.0


# predict when button is clicked
if st.sidebar.button("Predict"):
    #predict the result
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array) [0][1] * 100 #probability of fraud
    result = "🚨 Fraudulent Transaction" if prediction[0] == 1 else "legitimate Transaction"

#Display result
st.write(f"prediction: **{result}**")
st.write(f"Fraud Probability: **{probability:.2f}%**")

#show fraud probability visualization
fig, ax = plt.subplots()
ax.bar(["Legitimate", "Fraudulent"], [100 - probability, probability], color =["green", "red"])
ax.set_ylabel("probability (%)")
st.pyplot(fig)


#Custom footer
st.markdown("""
    <footer style="position: fixed; bottom: 10px; width: 100%; text-align: center; font-size: 12px; color: gray;">
        <p>FraudInspectAI © 2025 | Designed to help financial institutions combat fraud| @University of Wolverhampton|model</p>
    </footer>
""", unsafe_allow_html=True)

# In[13]:

