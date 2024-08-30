import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv('final_df.csv')
model = joblib.load("random_forest.pkl")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if 'result' not in st.session_state:
    st.session_state.result = None

st.set_page_config("Units Price Predictions", layout='wide')

def predict(total_sqft,bath,balcony,bhk, price_per_sqft, availability, standardized_location):    
    location_encode_v = None
    availability_binary = None
    clean_standardized_location = standardized_location

    matched_row = df[df['standardized_location'] == clean_standardized_location].head(1)

    if not matched_row.empty:
        location_encode_v =  matched_row.location_encoded
    else:
        location_encode_v  = None

    if availability == 'Ready To Move':
        availability_binary = 1 
    else: 
        availability_binary = 0
    x = np.zeros(7)
    x[0] = total_sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk
    x[4] = price_per_sqft
    x[5] = availability_binary
    x[6] = location_encode_v

    pred = model.predict([x])[0]

    # Store the result and switch to the Plots page
    st.session_state.result = round(pred, 2)
    st.session_state.page = 'Plots'


    
  
    
    
def home_page():
    st.title("Indian Real Estate Advisor")
    st.image("https://cdn-real-estate-egypt.coldwellbanker-eg.com/properties-3870/594678.jpg", width=1200)
    st.header("Objective")
    st.markdown("Providing insights for foriegn students for apartment prices according the specs they require")
    st.header("Data Sample for your inputs")
    st.dataframe(df[['total_sqft', 'bath', 'balcony', 'bhk', 'price_per_sqft', 'standardized_location']].head())

def inputs():
    st.title("Predict")

    total_sqft = st.number_input("Total sqft")
    bhk = st.number_input("Number of Bedrooms")
    nbath = st.number_input("Number of Baths")
    nbalcony = st.number_input("Number of Balcony")
    availability_status = st.radio("Availability", ['Ready to Move', 'Needs a While'])
    location = st.selectbox("Select the Area", df['standardized_location'].unique())
    price_per_sqft = st.number_input("Expected Price per sqft (30- 300)")

    st.button("Predict", on_click=predict, args=(total_sqft, nbath,nbalcony, bhk, price_per_sqft, availability_status, location ))

def plots(result = 0):
    st.title("Result")
    st.markdown(f"Estimated Fare: **{st.session_state.result}**") 

    fig = plt.figure()
    sns.histplot(df, x='price')
    st.pyplot(fig)

    if st.button("Make Another Prediction"):
        st.session_state.page = 'Predict'

def clear_cache_on_page_switch():
    if 'last_page' in st.session_state:
        if st.session_state.page != st.session_state.last_page:
            st.cache_data.clear()  # Clear data cache
            st.cache_resource.clear()  # Clear resource cache
    st.session_state.last_page = st.session_state.page

# Sidebar for page navigation
page = st.sidebar.selectbox("Select page", ["Home", "Predict", "Plots"], index=["Home", "Predict", "Plots"].index(st.session_state.page))

# # Update the page based on user selection in sidebar
if page:
    st.session_state.page = page

# Clear cache when switching pages
clear_cache_on_page_switch()

# Display the selected page
if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'Predict':
    inputs()
elif st.session_state.page == 'Plots':
    plots()
    

# # Sidebar for page navigation
# page = st.sidebar.selectbox("Select page", ["Home", "Predict", "Plots"])



# Display the selected page
# if st.session_state.page == 'Home':
#     home_page()
# elif st.session_state.page == 'Predict':
#     inputs()
# elif st.session_state.page == 'Plots':
#     plots()












