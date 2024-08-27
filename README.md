🔗Link: https://energyconsumptionforecasting.streamlit.app/
# Energy Consumption Forecasting

## Overview

This project focuses on forecasting energy consumption using machine learning techniques, specifically the Prophet model for time series forecasting. Additionally, it features a chatbot designed to provide answers and guidance on energy efficiency. The project is built with Python and deployed as a Streamlit app, offering an interactive platform for users to input data and receive forecasts along with energy efficiency advice.

## Features

- **Time Series Forecasting**: Utilizes the Prophet model to predict future energy consumption based on historical data and various factors.
- **Interactive Chatbot**: Provides information and answers to queries related to energy efficiency and usage.
- **User Input**: Allows users to input parameters such as temperature and working days to adjust predictions.
- **Visualization**: Includes graphical representations of past and predicted energy consumption trends.
- **Streamlit Dashboard**: Offers an interactive and user-friendly interface for interacting with the forecasting model and chatbot.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/energy-consumption-forecasting.git
   cd energy-consumption-forecasting
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv_name
   source venv_name/bin/activate  # On Windows, use `venv_name\Scripts\activate`
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

## Usage

1. **Start the App**: Run the Streamlit app as described above.
2. **Input Data**: Enter parameters like temperature and working days to adjust the forecasting model.
3. **View Forecasts**: Analyze the predicted energy consumption displayed on the dashboard.
4. **Interact with Chatbot**: Use the chatbot feature to get answers and tips on energy efficiency and how to use energy more effectively.

## Project Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of Python dependencies.
- `data/`: Directory containing datasets used for forecasting.
- `models/`: Directory for storing trained models.
- `scripts/`: Contains utility scripts for data processing, forecasting, and chatbot functionalities.
- `README.md`: This file.

## Dependencies

- Python 3.x
- Streamlit
- Prophet
- Pandas
- NumPy
- Matplotlib
- FAISS (for vector search)
- LangChain (for chatbot functionality)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, open an issue to discuss your proposed modifications.



---

Feel free to adjust or expand upon any section based on your project's specifics!
