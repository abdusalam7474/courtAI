import streamlit as st
import pandas as pd
import random
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# (Replace with your actual machine learning model)
def predict_intrusion(data, model):
    df = pd.DataFrame.from_dict(data, orient='index')
    #df = df.transpose()
    pred = model.predict(df)
    if pred[0] == 1:
        predicted_category = "Normal"
    else:
        predicted_category = "Possible Attack"
    return predicted_category, df

def preset_inputs(dfc):
  random_selections = {}
  # Get a random index
  r_i = random.randint(0, len(dfc)-1)
  # Extract the item
  random_item = dfc[(r_i-1):r_i]
  return random_item

dfz = pd.read_csv("my_data2.csv")
preset = preset_inputs(dfz)

# Data structure to hold user input (replace with actual feature names)
user_input = {
  0: {
    "MI_dir_L0.1_weight": None,
    "H_L0.1_weight": None,
    "HH_L0.1_weight": None,
    "HH_L0.1_std": None,
    "HH_L0.1_radius": None,
    "HH_L0.1_covariance": None,
    "HH_L0.1_pcc": None,
    "HH_jit_L0.1_weight": None,
    "HH_jit_L0.1_mean": None,
    "HpHp_L0.1_std": None,
    "HpHp_L0.1_radius": None,
    "HpHp_L0.1_covariance": None,
    "HpHp_L0.1_pcc": None,
    "Attack": None,
    "Attack_subType": None,
  }
}

loaded_rf = joblib.load('rf_model.pkl')
loaded_svm = joblib.load('svm_model.pkl')
loaded_dt = joblib.load('dtree_model.pkl')
models = {
    "Random forest":loaded_rf,
    "Support Vector Machine":loaded_svm,
    "Decision Tree":loaded_dt,
}

st.set_page_config(page_title="IoT Intrusion Detection", layout="wide")

# Sidebar for user convenience
with st.sidebar:
    st.header("Settings")
    # Add options for model selection, data preprocessing, etc. (if applicable)
    model_name = st.selectbox("select the model for prediction", ["Random forest", "Support Vector Machine", "Decision Tree"])
    selected_model = models[model_name]

st.title("IoT Intrusion Detection App")

# Overview section
st.markdown(
    """
    This app is built to help detect potential intrusions in an IoT network by analyzing data from your devices. It utilizes a machine learning model we trained to classify various readings as normal or indicative of an intrusion attempt.

    **How to Use:**

    1. Enter or or use the prefilled valuus (Prefilled values are randomly generated from the test set of the original dataset).
    2. Sellect your model of choice from the left sidebar.
    3. Click the "Predict" button.
    4. The app will display the predicted category ("Normal" or "Intrusion") and the provided sensor readings in a DataFrame.

    **Disclaimer:** This app is for demonstration purposes only. The accuracy of the predictions depends on the quality of the underlying machine learning model and sensor data. For real-world security applications, consult with security professionals.
    """
)

# Input section
st.header("Sensor Readings")
col1, col2, col3 = st.columns(3)
with col1:
    user_input[0]["MI_dir_L0.1_weight"] = st.number_input("MI_dir_L0.1_weight", value=(preset["MI_dir_L0.1_weight"]).iloc[0], format="%.15f")
    user_input[0]["H_L0.1_weight"] = st.number_input("H_L0.1_weight", value=(preset["H_L0.1_weight"]).iloc[0], format="%.15f")
    user_input[0]["HH_L0.1_pcc"] = st.number_input("HH_L0.1_pcc", value=(preset["HH_L0.1_pcc"]).iloc[0], format="%.15f")
    user_input[0]["HH_jit_L0.1_weight"] = st.number_input("HH_jit_L0.1_weight", value=(preset["HH_jit_L0.1_weight"]).iloc[0], format="%.15f")
with col2:
    user_input[0]["HH_L0.1_weight"] = st.number_input("HH_L0.1_weight", value=(preset["HH_L0.1_weight"]).iloc[0], format="%.15f")
    user_input[0]["HH_L0.1_std"] = st.number_input("HH_L0.1_std", value=(preset["HH_L0.1_std"]).iloc[0], format="%.15f")
    user_input[0]["HH_jit_L0.1_mean"] = st.number_input("HH_jit_L0.1_mean", value=(preset["HH_jit_L0.1_mean"]).iloc[0], format="%.15f")
    user_input[0]["HpHp_L0.1_std"] = st.number_input("HpHp_L0.1_std", value=(preset["HpHp_L0.1_std"]).iloc[0], format="%.15f")
with col3:
    user_input[0]["HH_L0.1_radius"] = st.number_input("HH_L0.1_radius", value=(preset["HH_L0.1_radius"]).iloc[0], format="%.15f")
    user_input[0]["HH_L0.1_covariance"] = st.number_input("HH_L0.1_covariance", value=(preset["HH_L0.1_covariance"]).iloc[0], format="%.15f")
    user_input[0]["HpHp_L0.1_radius"] = st.number_input("HpHp_L0.1_radius", value=(preset["HpHp_L0.1_radius"]).iloc[0], format="%.15f")
    user_input[0]["HpHp_L0.1_covariance"] = st.number_input("HpHp_L0.1_covariance", value=(preset["HpHp_L0.1_covariance"]).iloc[0], format="%.15f")
st.markdown(
    """
    **mapping for attack subtypes:** {'combo': 0, 'junk': 1, 'scan': 2, 'tcp': 3, 'udp': 4, 'ack': 5, 'syn': 6, 'udpplain': 7, 'Normal': 8}.
    """
)
st.markdown(
    """    
    **mapping for attack types:** {'gafgyt': 0, 'mirai': 1, 'Normal': 2}.
    """
)
col1, col2 = st.columns(2)
with col1:
    user_input[0]["Attack"] = st.number_input("Attack", value=(preset["Attack"]).iloc[0], min_value=(0), max_value=(2))
    user_input[0]["HpHp_L0.1_pcc"] = st.number_input("HpHp_L0.1_pcc", value=(preset["HpHp_L0.1_pcc"]).iloc[0], format="%.15f")
with col2:
    user_input[0]["Attack_subType"] = st.number_input("Attack_subType", value=(preset["Attack_subType"]).iloc[0], min_value=(0), max_value=(8))

# Prediction button and results section
predict_button = st.button("Predict")

if predict_button:
    predicted_category, data_df = predict_intrusion(user_input, selected_model)
    st.subheader("Prediction Results")
    st.dataframe(data_df)
    st.write("Predicted Category:", predicted_category)
