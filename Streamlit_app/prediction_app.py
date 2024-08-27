# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import streamlit as st
# import pickle
# from datetime import datetime, timedelta

# def run_energy_forecast_app():
#     # Function to calculate 'summer_temp' based on temperature
#     def summer_temp(temp):
#         return 1 if temp > 20 else 0

#     # Function to categorize the month into bins
#     def categorize_month(month):
#         return pd.cut([month], bins=3, labels=False)[0]

#     # Custom CSS for black theme with enhancements
#     st.markdown("""
#     <style>
#         .reportview-container {
#             background: linear-gradient(135deg, #1c1e24 10%, #23262d 100%);
#             color: #fafafa;
#         }
#         .main {
#             background: #2e3038;
#             padding: 2rem;
#             border-radius: 10px;
#             box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
#         }
#         h1, h2, h3, .stButton>button {
#             color: #fafafa;
#         }
#         .stButton>button {
#             background-color: #4e8cff;
#             color: white;
#             border-radius: 8px;
#             padding: 0.5rem 1rem;
#             font-weight: bold;
#         }
#         .stTextInput>div>div>input {
#             background-color: #3a3d46;
#             color: #fafafa;
#             border: 1px solid #4e8cff;
#             border-radius: 8px;
#         }
#     </style>
#     """, unsafe_allow_html=True)

#     # Streamlit app setup
#     st.title('💡 Energy Consumption Forecast')

#     # Load the saved Prophet model
#     @st.cache_resource
#     def load_model():
#         with open("..\Streamlit_app\prophet_model.pkl", 'rb') as f:
#             return pickle.load(f)

#     loaded_model = load_model()

#     # User inputs section
#     st.header('📅 Input Data')

#     # Date input with a broader date range and calendar view
#     min_date = datetime.now().date() - timedelta(days=365*10) 
#     max_date = datetime.now().date() + timedelta(days=365*10) 
#     start_date_input = st.date_input("Select the start date:", min_value=min_date, max_value=max_date)
#     start_date = pd.to_datetime(start_date_input)

#     # Input layout with better spacing and aesthetics
#     st.subheader('📊 Enter Forecast Data')

#     # User data input loop with streamlined design
#     user_data_list = []
#     for i in range(5):
#         next_date = start_date + pd.Timedelta(days=i)
        
#         with st.expander(f"Day {i+1}: {next_date.date()}"):
#             temp_input = st.number_input(f"Temperature (°C):", min_value=-30.0, max_value=50.0, step=0.1, key=f"temp_{i}")
#             is_working_day_input = st.selectbox(f"Working day?", ('Yes', 'No'), key=f"work_{i}")
            
#             # Convert inputs
#             temperature = temp_input
#             is_working_day = 1 if is_working_day_input == 'Yes' else 0
#             month = next_date.month
            
#             # Calculate additional features
#             summer_temp_value = summer_temp(temperature)
#             month_bin_value = categorize_month(month)
            
#             # Add data for this day to the list
#             user_data_list.append({
#                 'ds': next_date,
#                 'temp': temperature,
#                 'summer_temp': summer_temp_value,
#                 'month_bins': month_bin_value,
#                 'Is_Working_Day': is_working_day
#             })

#     # Convert list to DataFrame
#     user_data = pd.DataFrame(user_data_list)

#     # Make predictions
#     forecast = loaded_model.predict(user_data)

#     # Display predictions
#     st.header('🚀 5-Day Forecast')
#     st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].style.format({'yhat': '{:.2f}', 'yhat_lower': '{:.2f}', 'yhat_upper': '{:.2f}'}))

#     # Visualization section
#     st.header('📊 Forecast Visualization')

#     # Plot 1: Forecast with confidence interval
#     fig1 = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], 
#                    labels={'ds': 'Date', 'value': 'Energy Consumption', 'variable': 'Forecast'},
#                    title='5-Day Energy Consumption Forecast')
#     fig1.update_layout(legend_title_text='', template="plotly_dark")
#     st.plotly_chart(fig1)

#     # Plot 2: Factor heatmap with predicted consumption
#     heatmap_data = user_data.copy()
#     heatmap_data['Predicted Consumption'] = forecast['yhat']
#     fig2 = px.imshow(heatmap_data[['temp', 'Is_Working_Day', 'Predicted Consumption']].T,
#                      labels=dict(x="Date", y="Factor", color="Value"),
#                      x=heatmap_data['ds'].dt.date,
#                      y=['Temperature', 'Is Working Day', 'Predicted Consumption'],
#                      title="Factor Heatmap and Predicted Consumption")
#     fig2.update_layout(template="plotly_dark")
#     st.plotly_chart(fig2)

#     # Plot 3: Working vs Non-Working Days and Temperature Distribution
#     fig3 = px.scatter(user_data, x='ds', y='temp', color='Is_Working_Day',
#                       labels={'ds': 'Date', 'temp': 'Temperature (°C)', 'Is_Working_Day': 'Working Day'},
#                       title='Temperature and Working Day Distribution',
#                       color_discrete_map={0: 'red', 1: 'green'})
#     fig3.update_layout(template="plotly_dark")
#     st.plotly_chart(fig3)

#     # Download forecast as CSV
#     st.subheader('📥 Download Forecast')
#     csv = forecast.to_csv(index=False)
#     st.download_button(
#         label="Download forecast as CSV",
#         data=csv,
#         file_name="energy_forecast.csv",
#         mime="text/csv",
#     )

# # To run the app from the same file
# # if __name__ == "__main__":
# #     run_energy_forecast_app()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pickle
from datetime import datetime, timedelta

def run_energy_forecast_app():
    # Function to calculate 'summer_temp' based on temperature
    def summer_temp(temp):
        return 1 if temp > 20 else 0

    # Function to categorize the month into bins
    def categorize_month(month):
        return pd.cut([month], bins=3, labels=False)[0]

    # Custom CSS for a single color theme
    st.markdown("""
        <style>
            .reportview-container {
                background: #000000; /* Black background */
                color: #ffffff; /* Bright white text */
            }
            .main {
                background: #000000; /* Black background */
                color: #ffffff; /* Bright white text */
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            }
            h1, h2, h3, .stButton>button {
                color: #ffffff; /* Bright white text */
            }
            .stButton>button {
                background-color: #444444; /* Slightly lighter dark gray for buttons */
                color: #ffffff; /* Bright white text */
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }
            .stTextInput>div>div>input {
                background-color: #222222; /* Dark gray background for inputs */
                color: #ffffff; /* Bright white text */
                border: 1px solid #ffffff; /* Bright white border */
                border-radius: 8px;
            }
        </style>
        """, unsafe_allow_html=True)




    # Streamlit app setup
    st.title('💡 Energy Consumption Forecast')

    # Load the saved Prophet model
    @st.cache_resource
    def load_model():
        try:
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, 'prophet_model.pkl')
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error("Model file not found. Please check the path.")
            return None
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
            return None

    loaded_model = load_model()
    if loaded_model is None:
        return

    # User inputs section
    st.header('📅 Input Data')

    # Date input with a broader date range and calendar view
    min_date = datetime.now().date() - timedelta(days=365*10)
    max_date = datetime.now().date() + timedelta(days=365*10)
    start_date_input = st.date_input("Select the start date:", min_value=min_date, max_value=max_date)
    start_date = pd.to_datetime(start_date_input)

    # Input layout with better spacing and aesthetics
    st.subheader('📊 Enter Forecast Data')

    # User data input loop with streamlined design
    user_data_list = []
    for i in range(5):
        next_date = start_date + pd.Timedelta(days=i)
        
        with st.expander(f"Day {i+1}: {next_date.date()}"):
            temp_input = st.number_input(f"Temperature (°C):", min_value=-30.0, max_value=50.0, step=0.1, key=f"temp_{i}")
            is_working_day_input = st.selectbox(f"Working day?", ('Yes', 'No'), key=f"work_{i}")
            
            # Convert inputs
            temperature = temp_input
            is_working_day = 1 if is_working_day_input == 'Yes' else 0
            month = next_date.month
            
            # Calculate additional features
            summer_temp_value = summer_temp(temperature)
            month_bin_value = categorize_month(month)
            
            # Add data for this day to the list
            user_data_list.append({
                'ds': next_date,
                'temp': temperature,
                'summer_temp': summer_temp_value,
                'month_bins': month_bin_value,
                'Is_Working_Day': is_working_day
            })

    # Convert list to DataFrame
    user_data = pd.DataFrame(user_data_list)

    # Make predictions
    forecast = loaded_model.predict(user_data)

    # Rename columns in forecast DataFrame
    forecast = forecast.rename(columns={
        'ds': 'Date',
        'yhat': 'Energy Usage',
        'yhat_lower': 'Minimum Energy Usage',
        'yhat_upper': 'Maximum Energy Usage'
    })

    # Display predictions
    st.header('🚀 5-Day Forecast')
    st.dataframe(forecast[['Date', 'Energy Usage', 'Minimum Energy Usage', 'Maximum Energy Usage']].style.format({'Energy Usage': '{:.2f}', 'Minimum Energy Usage': '{:.2f}', 'Maximum Energy Usage': '{:.2f}'}))

    # Visualization section
    st.header('📊 Forecast Visualization')

    # Plot 1: Forecast with confidence interval
    fig1 = px.line(forecast, x='Date', y=['Energy Usage', 'Minimum Energy Usage', 'Maximum Energy Usage'], 
                   labels={'Date': 'Date', 'value': 'Energy Consumption', 'variable': 'Forecast'},
                   title='5-Day Energy Consumption Forecast')
    fig1.update_layout(legend_title_text='', template="plotly_dark")
    st.plotly_chart(fig1)

    # Plot 2: Factor heatmap with predicted consumption
    heatmap_data = user_data.copy()
    heatmap_data['Predicted Consumption'] = forecast['Energy Usage']
    fig2 = px.imshow(heatmap_data[['temp', 'Is_Working_Day', 'Predicted Consumption']].T,
                     labels=dict(x="Date", y="Factor", color="Value"),
                     x=heatmap_data['ds'].dt.date,
                     y=['Temperature', 'Is Working Day', 'Predicted Consumption'],
                     title="Factor Heatmap and Predicted Consumption")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2)

    # Plot 3: Working vs Non-Working Days and Temperature Distribution
    fig3 = px.scatter(user_data, x='ds', y='temp', color='Is_Working_Day',
                      labels={'ds': 'Date', 'temp': 'Temperature (°C)', 'Is_Working_Day': 'Working Day'},
                      title='Temperature and Working Day Distribution',
                      color_discrete_map={0: 'red', 1: 'green'})
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3)

    # Download forecast as CSV
    st.subheader('📥 Download Forecast')
    csv = forecast.to_csv(index=False)
    st.download_button(
        label="Download forecast as CSV",
        data=csv,
        file_name="energy_forecast.csv",
        mime="text/csv",
    )

# Uncomment to run the app directly
if __name__ == "__main__":
    run_energy_forecast_app()
