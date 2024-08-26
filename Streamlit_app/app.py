import streamlit as st
from prediction_app import run_energy_forecast_app
from Energy_consumption_chatbot import run_energy_qa_app
import time

def main():
    # Set page config at the very start of the main script
    st.set_page_config(page_title="Energy Management System", page_icon="⚡", layout="wide")

    # Sidebar for navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['📈 Forecasting', '💬 Chatbot', '❗ About'])

    # Show the selected page with a buffer for loading
    with st.spinner('Loading...'):
        time.sleep(1)  # Adjust the buffer time if needed
        
        if page == '📈 Forecasting':
            run_energy_forecast_app()
        elif page == '💬 Chatbot':
            run_energy_qa_app()
        elif page == '❗ About':
            show_about_page()

def show_about_page():
    st.title('About This Application')

    st.header('Energy Consumption Forecasting')
    st.subheader('Prophet Model')
    st.write("""
        The Prophet model, developed by Facebook, is used for time series forecasting. 
        It is particularly well-suited for data with strong seasonal patterns and 
        historical data with missing values or irregularities. Prophet decomposes 
        the time series into trend, seasonality, and holidays to make accurate predictions.
        It is robust to missing data and shifts in the trend, making it ideal for energy 
        consumption forecasting.
    """)

    st.header('Energy Consumption Chatbot')
    st.subheader('Chatbot Details')
    st.write("""
        The Energy Consumption Chatbot utilizes natural language processing to provide 
        answers and insights about energy usage and efficiency. It is designed to help 
        users understand their energy consumption patterns, optimize usage, and answer 
        frequently asked questions about energy management. The chatbot is built using 
        state-of-the-art language models and is integrated with a retrieval-augmented 
        generation (RAG) architecture for accurate and contextually relevant responses.
    """)

if __name__ == "__main__":
    main()
