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
# st.sidebar.subheader("Subheading")
# st.sidebar.text("Sidebar content goes here.")

# PULocationID,DOLocationID,is_holiday,hour,week_day,month,request_time_group,weather_condition_code

# Location: PULocationID,DOLocationID
taxi_zone_df = pd.read_csv('sdata/taxi_zone_lookup.csv')

pickup_location = st.selectbox('PickUp Location', taxi_zone_df['Zone'])
PULocationID = taxi_zone_df.loc[taxi_zone_df['Zone'] == pickup_location, 'LocationID'].values[0]
#st.write(f"Selected Zone: {pickup_location}. Corresponding LocationID: {PULocationID}")

dropoff_location = st.selectbox('DropOff Location', taxi_zone_df['Zone'])
DOLocationID = taxi_zone_df.loc[taxi_zone_df['Zone'] == dropoff_location, 'LocationID'].values[0]
#st.write(f"Selected Zone: {dropoff_location}. Corresponding LocationID: {DOLocationID}")

# Create a date picker
selected_date = st.date_input("Select a date", value="today")
#st.write(f"PickUp Date: {selected_date}")

#week_day
week_day = selected_date.weekday()
#st.write('The week_day given is ', week_day)

#month
month = selected_date.month
#st.write('The month given is ', month)

#Load holiday data
holiday_data = pd.read_csv('sdata/nyc_holiday.csv')
#return 0 or 1 if selected_date is a holiday
selected_date_pd = pd.to_datetime(selected_date)
holiday_data['date'] = pd.to_datetime(holiday_data['date'])
is_holiday = selected_date_pd in holiday_data['date'].values
holiday_name = ''
if is_holiday:
    holiday_name = holiday_data[holiday_data['date'] == selected_date_pd]['holiday_name'].values[0]

#st.write(f"Selected Holiday: {is_holiday}. Corresponding IsHoliday: {holiday_name}")


# Display a time input widget
selected_time = st.time_input("Select a time", value="now")
#st.write("You selected:", selected_time)

#hour
hour = selected_time.hour
#st.write('The Hour given is ', hour)

# Load in weather condition data
weather_data = pd.read_csv('sdata/weather_condition.csv')
weather_condition = st.selectbox('Weather Condition', weather_data['Condition'])
weather_condition_code = weather_data.loc[weather_data['Condition'] == weather_condition, 'Code'].values[0]
#st.write(f"Selected Condition: {weather_condition}. Corresponding Code: {weather_condition_code}")

predict_button = st.button("Predict")

if predict_button:

    # Initialize the default group as Night (0)
    request_time_group = 0

    # Define conditions for Peak (2) and OffPeak (1) times for weekday
    weekday_mask = week_day < 5

    peak_mask = (hour >= 6) & (hour < 10) | (hour >= 15) & (hour < 19)
    off_peak_mask = (hour >= 10) & (hour < 15) | (hour >= 19) & (hour < 22)

    # For weekend
    weekend_mask = ~weekday_mask

    # Apply conditions for weekdays

    # If day is not holiday, it will be Peak
    if weekday_mask & peak_mask & ~is_holiday:
        request_time_group = 2
    # If day is holiday, it will be Off-Peak
    if weekday_mask & peak_mask & is_holiday: 
        request_time_group = 1

    if weekday_mask & off_peak_mask:
        request_time_group = 1

    # Apply conditions for weekends
    if weekend_mask & (hour >= 6) & (hour < 22):
        request_time_group = 1

    st.write(PULocationID, DOLocationID, int(is_holiday), hour, week_day, month, request_time_group, weather_condition_code)
    result = loaded_model.predict([[PULocationID, DOLocationID, int(is_holiday), hour, week_day, month,request_time_group, weather_condition_code]])

    #convert result which is in seconds to minutes
    result = int(result)/60
    result = round(result, 2)
    st.write("The Predicted Trip Time is {} minutes.".format(result))


st.write('Thank you for Trying out our App')
st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")



#connect model