#import libraries

import pandas as pd
import numpy as np
import streamlit as st
import pickle


#load in model
filename = 'lr_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))


#build a simple streamlit app
st.set_page_config(layout="wide")
st.header('Travel Time Predictor App')

# Custom HTML/CSS for the banner
custom_html = """
<div class="banner">
    <img src="https://img.freepik.com/premium-photo/wide-banner-with-many-random-square-hexagons-charcoal-dark-black-color_105589-1820.jpg" alt="Banner Image">
</div>
<style>
    .banner {
        width: 160%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

# Sidebar content
st.sidebar.subheader("Subheading")
st.sidebar.text("Sidebar content goes here.")

#PULocationID,DOLocationID,is_holiday,hour,week_day,month,request_time_group,weather_condition_code

#Location
taxi_zone_df = pd.read_csv('data/taxi_zone_lookup.csv')

pickup_location = st.selectbox('PickUp Location', taxi_zone_df['Zone'])
PULocationID = taxi_zone_df.loc[taxi_zone_df['Zone'] == pickup_location, 'LocationID'].values[0]
st.write(f"Selected Zone: {pickup_location}. Corresponding LocationID: {PULocationID}")

dropoff_location = st.selectbox('DropOff Location', taxi_zone_df['Zone'])
DOLocationID = taxi_zone_df.loc[taxi_zone_df['Zone'] == dropoff_location, 'LocationID'].values[0]
st.write(f"Selected Zone: {dropoff_location}. Corresponding LocationID: {DOLocationID}")

# Create a date picker
selected_date = st.date_input("Select a date", value="today")
st.write(f"PickUp Date: {selected_date}")

# Display a time input widget
selected_time = st.time_input("Select a time", value="now")
st.write("You selected:", selected_time)


height = st.number_input('Height')
st.write('The Height given is ', height)


sex = st.selectbox(
    'Sex',
    ('Male', 'Female'))

st.write('You selected:', sex)


predict_button = st.button("Predict")

if predict_button:
    #sex_no 1 = Male
    #sex_no 2 = Female
    if sex == 'Male':
        sex_no = 1
    else:
        sex_no = 2

    result = loaded_model.predict([[height,sex_no]])

    st.write("The Predicted Shoe-Size is {}".format(int(result)))

    #give feedback
    feedback = st.selectbox(
        'Is our prediction Right or Wrong',
        ('Right','Wrong')
    )


st.write('Thank you for Trying out our App')
st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")


#connect model