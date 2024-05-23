import streamlit as st
import pandas as pd
import random
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Court Case Prediction", layout="wide")

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

def predict_intrusion_(data, model):
    df = pd.DataFrame.from_dict(data, orient='index')
    #df = df.transpose()
    #pred = model.predict(df)
    pred = [0,1]
    if pred[0] == 1:
        predicted_category = "Normal"
    else:
        predicted_category = "Possible Attack"
    return predicted_category, df
    
@st.cache_resource
def preset_inputs(dfc):
  random_selections = {}
  # Get a random index
  r_i = random.randint(0, len(dfc)-1)
  # Extract the item
  random_item = dfc[(r_i-1):r_i]
  return random_item

@st.cache_resource
def load_data():
    dfz = pd.read_csv("Texas_Department_dataset.csv")
    #coverting date to date-time objects
    dfz["Release Date"] = pd.to_datetime(dfz["Release Date"])
    dfz["Sentence Date"] = pd.to_datetime(dfz["Sentence Date"])
    dfz["Offense Date"] = pd.to_datetime(dfz["Offense Date"])
    
    return dfz

dfz = load_data()
preset = preset_inputs(dfz)

# Data structure to hold user input (replace with actual feature names)
user_input = {
  0: {
    "Release Date": None,
    "Inmate Type": None,
    "Gender": None,
    "Race": None,    
    "Age": None,
    "County": None,
    "Offense": None,
    "Offense Description": None,
    "Sentence Date": None,
    "Offense Date": None,
    #"Sentence (Years)": None,
  }
}

@st.cache_resource
def load_models():
   loaded_rf = joblib.load('rf_model.pkl')
   loaded_svm = joblib.load('svm_model.pkl')
   loaded_dt = joblib.load('dtree_model.pkl')
   models = {
    "Random forest":loaded_rf,
    "Support Vector Machine":loaded_svm,
    "Decision Tree":loaded_dt,
   }
   return models

models = load_models()
theft_or_larc_opts = [(preset["Offense"]).iloc[0],"THEFT PROPERTY", "LARCENCY-THEFT OF CREDIT CARD", "THEFT OF FIREARM", "LARCENCY THEFT OF PERSON", "STOLEN VEHICLE THEFT", "THEFT FROM PERSON", "THEFT OF SERVICE", "LARCENSY THEFT OF PROPERTY", "THEFT OF MATERIAL ALUMINUM or BRONZE or COPPER or BRASS"]
amount_opts = ["less than 1,500", "less than 2,500", "greater than or equal to 2,500, less than 30K", "greater than or equal to 20K less than 100k", "greater than 200k", "greater than or equal to 30K, less than 150k", "greater than or equal to 1,500, less than 20K", "less than 20K"]
race_opts = [(preset["Race"]).iloc[0], 'White', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaskin', 'Other']
inmate_opts = [(preset["Inmate Type"]).iloc[0], 'G2', 'FT', 'J2', 'DP', 'IT', 'OT', 'J1', 'MD', 'G1', 'G4', 'S1', 'PR', 'P2', 'G5', 'RP', 'MH', 'J5', 'J4', 'VI', 'PS', 'RF', 'PJ', '1A', 'XX', 'CG', 'CP', 'II', 'P4']
offense_opts = [(preset["Offense"]).iloc[0]]
gender_opts = [(preset["Gender"]).iloc[0], 'M', 'F']
#age_opts
county_opts = [(preset["County"]).iloc[0], 'Houston', 'Dallas', 'Victoria', 'Angelina', 'Hopkins', 'Travis', 'Johnson', 'Ellis', 'Montgomery', 'Jefferson', 'Fannin', 'Guadalupe', 'Taylor', 'Nueces', 'Harris', 'Bexar', 'Ector', 'Galveston', 'Denton', 'Midland', 'Tarrant', 'Williamson', 'Potter', 'Lubbock', 'Rockwall', 'Leon', 'Bowie', 'San Patricio', 'Hays', 'Nacogdoches', 'Smith', 'Lamar', 'Navarro', 'Wharton', 'Hidalgo', 'Hood']


# Sidebar for user convenience
with st.sidebar:
    st.header("Settings")
    # Add options for model selection, data preprocessing, etc. (if applicable)
    model_name = st.selectbox("select the model for prediction", ["Random forest", "Support Vector Machine", "Decision Tree"])
    selected_model = models[model_name]

st.title("Court Case Prediction App")

# Overview section
st.markdown(
    """
    This app is built to help predict rullings of court cases by predicting the potential release type of a convicted individual. It utilizes a machine learning model we trained to classify cases into a potential release type.

    **How to Use:**

    1. Enter or or use the prefilled valuus (Prefilled values are randomly generated from the test set of the original dataset).
    2. Sellect your model of choice from the left sidebar.
    3. Click the "Predict" button.
    4. The app will display the predicted category ("Normal" or "Intrusion") and the provided sensor readings in a DataFrame.

    **Disclaimer:** This app is for demonstration purposes only. The accuracy of the predictions depends on the quality of the underlying machine learning model and sensor data. For real-world legal applications, consult with legal professionals.
    """
)

# Input section
st.header("Case Details")
col1, col2, col3 = st.columns(3)
with col1:
    user_input[0]["Race"] = st.selectbox("Race", race_opts)
    user_input[0][" Inmate Type"] = st.selectbox(" Inmate Type", inmate_opts)
    user_input[0]["Offense Date"] = st.date_input("Offense Date", value=(preset["Offense Date"]).iloc[0], format="DD/MM/YYYY")
with col2:
    user_input[0]["Gender"] = st.selectbox("Gender", gender_opts)
    user_input[0]["Age"] = st.number_input("Age", value=(preset["Age"]).iloc[0])
    user_input[0]["Sentence Date"] = st.date_input("Sentence Date", value=(preset["Sentence Date"]).iloc[0], format="DD/MM/YYYY")
    #user_input[0]["Sentence (Years)"] = st.selectbox("Sentence (Years)", value=(preset["Sentence (Years)"]).iloc[0])
with col3:
    user_input[0]["County"] = st.selectbox("County", county_opts)
    user_input[0]["Offense"] = st.selectbox("Offense", offense_opts)
    user_input[0]["Release Date"] = st.date_input("Release Date", value=(preset["Release Date"]).iloc[0], format="DD/MM/YYYY")

col1, col2, col3= st.columns(3)
with col1:
    theft_or_larcency = st.selectbox("THEFT OR LARCENCY DEFINITION", theft_or_larc_opts, key="theft_descr")
    #user_input[0]["Offense Description"] = st.selectbox("Offense Description", value=(preset["Offense Description"]).iloc[0])
with col2:
    amount = st.selectbox("Amount or Worth of Stolen Item ($)", amount_opts, key="amount_in_que")
with col3:
    prev_conv = st.selectbox("Previous Convictions (if any):", [None, "2 or more previous convictions"])
# Prediction button and results section
predict_button = st.button("Predict")

if predict_button:
    predicted_category, data_df = predict_intrusion(user_input, selected_model)
    st.subheader("Prediction Results")
    st.dataframe(data_df)
    st.write("Predicted Category:", predicted_category)
