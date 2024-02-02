import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from geopy.distance import geodesic
import requests
import statistics
import json


data = pd.read_csv('D:\Subhashini\Datascience\Tools\mrt.csv')
mrt_location = pd.DataFrame(data)

# Setting up page configuration
st.set_page_config(page_title="Singapore Resale Flat Prices Predicting",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Creating option menu in the sidebar
with st.sidebar:
    selected = st.radio("Menu", ["Home", "Resale Price"], index=0)

if selected == 'Home':
    st.title(":green[*SINGAPORE RESALE FLAT PRICES PREDICTING*]")

    col1, col2 = st.columns(2)

    with col1:
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :violet[*Overview*] : Build regression model to predict resale price")
        col1.markdown("# ")
        col1.markdown("## :violet[*Domain*] : Resale Flat Prices")
        col1.markdown("# ")
        col1.markdown("## :violet[*Technologies used*] : Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Streamlit.")

    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        image = Image.open(r"D:\Subhashini\Datascience\Tools\flat.jpg")
        st.image(image, caption='Image', use_column_width=True)
        col2.markdown("# ")


if selected == 'Resale Price':
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    try:
        with st.form("form1"):

            street_name_df = pd.read_csv('D:\Subhashini\Datascience\Tools\street_name.csv')

            # Street_Name

            street_name = st.selectbox('Street Name', street_name_df['street_name'])

            # Block Number

            block_no_df = pd.read_csv('D:\Subhashini\Datascience\Tools\block.csv')
            block = st.selectbox('Block Number', block_no_df['block'])

            # Floor Area

            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)  [Note: min_value=1.0, max_value=500.0]', min_value=1.0, max_value=500.0)
            

            # Lease Commence Date

            lease_commence_date = st.number_input('Lease Commence Date ')

            # Storey Range

            storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            # PREDICT RESALE PRICE

            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button is not None:
                with open(r"model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)

                # Calculating lease_remain_years using lease_commence_date
                    
                lease_remain_years = 99 - (2023 - lease_commence_date)

                # Calculating median of storey_range to make our calculations quite comfortable

                split_list = storey_range.split(' TO ')
                float_list = [float(i) for i in split_list]
                storey_median = statistics.median(float_list)

                # Getting the address by joining the block number and the street name

                address = block + " " + street_name
                query_address = address
                query_string = 'https://www.onemap.gov.sg/api/common/elastic/search?searchVal=' + str(
                    query_address) + '&returnGeom=Y&getAddrDetails=Y'
                resp = requests.get(query_string)

                # Using OpenMap API getting the latitude and longitude location of that address

                origin = []
                data_geo_location = json.loads(resp.content)
                if data_geo_location['found'] != 0:
                    latitude = data_geo_location['results'][0]['LATITUDE']
                    longitude = data_geo_location['results'][0]['LONGITUDE']
                    origin.append((latitude, longitude))

                # Appending the Latitudes and Longitudes of the MRT Stations
                    
                # Latitudes and Longitudes are been appended in the form of a tuple  to that list
                    
                mrt_lat = mrt_location['latitude']
                mrt_long = mrt_location['longitude']
                list_of_mrt_coordinates = []
                for lat, long in zip(mrt_lat, mrt_long):
                    list_of_mrt_coordinates.append((lat, long))

                # Getting distance to nearest MRT Stations (Mass Rapid Transit System)
                    
                list_of_dist_mrt = []
                for destination in range(0, len(list_of_mrt_coordinates)):
                    list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
                shortest = (min(list_of_dist_mrt))
                min_dist_mrt = shortest
                list_of_dist_mrt.clear()

                # Getting distance from CDB (Central Business District)

                cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                # Sending the user enter values for prediction to our model

                new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample[:, :5])
                new_pred = loaded_model.predict(new_sample)[0]
                st.markdown(f'## :violet[Predicted Resale Price:] <span style="color:green">{np.exp(new_pred)}</span>', unsafe_allow_html=True)

    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat")