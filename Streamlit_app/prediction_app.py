import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pickle
from datetime import datetime, timedelta
# import faiss
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Forecasting functions
def summer_temp(temp):
    return 1 if temp > 20 else 0

def categorize_month(month):
    return pd.cut([month], bins=3, labels=False)[0]

@st.cache_resource
def load_model():
    with open("..\Streamlit_app\prophet_model.pkl", 'rb') as f:
        return pickle.load(f)

# Q&A Bot functions
def load_components():
    index = FAISS.read_index(r"..\Streamlit_app\vector_store.index")
    
    with open("..\Streamlit_app\docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    
    with open("..\Streamlit_app\index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)
    
    with open("..\Streamlit_app\embedding.pkl", "rb") as f:
        embeddings = pickle.load(f)
    
    return index, docstore, index_to_docstore_id, embeddings

def create_vector_store(embeddings, index, docstore, index_to_docstore_id):
    return FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)

def init_language_model():
    return HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')

def create_prompt():
    prompt_template = """
    You are a Q&A assistant specializing in energy-related topics based on the content of a provided PDF.
    Your goal is to provide accurate and concise answers to questions specifically about energy consumption or related subjects covered in the PDF.

    1. **Answer the question** with a brief paragraph summarizing the relevant information from the PDF. Don't say you are answering from PDF but directly answer the question.
    2. **Explain the reason for your answer** in a second paragraph by referring to the specific content or details from the PDF that support your response.
    3. **If the question does not pertain to energy or the content of the PDF**, respond with: "This question is not related to the topics covered in the provided PDF."

    {context}

    Question: {input}
    Answer:
    """
    return ChatPromptTemplate.from_template(prompt_template)

def create_chains(llm, prompt, vector_store):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

def get_response(query, retrieval_chain, general_responses):
    normalized_query = query.lower().strip()
    
    if normalized_query in general_responses:
        return general_responses[normalized_query]
    else:
        response = retrieval_chain.invoke({"input": query})
        answer = response["answer"]
        
        answer_marker = "Answer:"
        start_index = answer.find(answer_marker)
        
        if start_index != -1:
            generated_output = answer[start_index + len(answer_marker):].strip()
            return "\n".join(line.strip() for line in generated_output.splitlines() if line.strip())
        else:
            return "Answer marker not found. Here is the raw response:\n" + answer.strip()

# Custom CSS
def set_custom_css():
    st.markdown("""
    <style>
        .reportview-container {
            background: linear-gradient(135deg, #1c1e24 10%, #23262d 100%);
            color: #fafafa;
        }
        .main {
            background: #2e3038;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        h1, h2, h3, .stButton>button {
            color: #fafafa;
        }
        .stButton>button {
            background-color: #4e8cff;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            background-color: #3a3d46;
            color: #fafafa;
            border: 1px solid #4e8cff;
            border-radius: 8px;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #2b313e;
            color: #ffffff;
            border-bottom-right-radius: 0;
            margin-left: 40%;
        }
        .chat-message.bot {
            background-color: #475063;
            color: #ffffff;
            border-bottom-left-radius: 0;
            margin-right: 40%;
        }
        .chat-message .avatar {
            width: 20%;
        }
        .chat-message .message {
            width: 80%;
            padding: 0 1.5rem;
        }
        .avatar-icon {
            font-size: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Page functions
def forecasting_page():
    st.title('💡 Energy Consumption Forecast')

    loaded_model = load_model()

    st.header('📅 Input Data')

    min_date = datetime.now().date() - timedelta(days=365*10)
    max_date = datetime.now().date() + timedelta(days=365*10)
    start_date_input = st.date_input("Select the start date:", min_value=min_date, max_value=max_date)
    start_date = pd.to_datetime(start_date_input)

    st.subheader('📊 Enter Forecast Data')

    user_data_list = []
    for i in range(5):
        next_date = start_date + pd.Timedelta(days=i)
        
        with st.expander(f"Day {i+1}: {next_date.date()}"):
            temp_input = st.number_input(f"Temperature (°C):", min_value=-30.0, max_value=50.0, step=0.1, key=f"temp_{i}")
            is_working_day_input = st.selectbox(f"Working day?", ('Yes', 'No'), key=f"work_{i}")
            
            temperature = temp_input
            is_working_day = 1 if is_working_day_input == 'Yes' else 0
            month = next_date.month
            
            summer_temp_value = summer_temp(temperature)
            month_bin_value = categorize_month(month)
            
            user_data_list.append({
                'ds': next_date,
                'temp': temperature,
                'summer_temp': summer_temp_value,
                'month_bins': month_bin_value,
                'Is_Working_Day': is_working_day
            })

    user_data = pd.DataFrame(user_data_list)
    forecast = loaded_model.predict(user_data)

    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={
            'ds': 'Date',
            'yhat': 'Energy Usage',
            'yhat_lower': 'Minimum Energy Usage',
            'yhat_upper': 'Maximum Energy Usage'
        }
    )

    st.header('🚀 5-Day Forecast')
    st.dataframe(forecast_display.style.format({
        'Energy Usage': '{:.2f}', 
        'Minimum Energy Usage': '{:.2f}', 
        'Maximum Energy Usage': '{:.2f}'
    }))

    st.header('✨ Forecast Visualization')

    fig1 = px.line(forecast_display, x='Date', y=['Energy Usage', 'Minimum Energy Usage', 'Maximum Energy Usage'],
                   labels={'Date': 'Date', 'value': 'Energy Consumption', 'variable': 'Forecast'},
                   title='5-Day Energy Consumption Forecast')
    fig1.update_layout(legend_title_text='', template="plotly_dark")
    st.plotly_chart(fig1)

    heatmap_data = user_data.copy()
    heatmap_data['Predicted Consumption'] = forecast['yhat']
    fig2 = px.imshow(heatmap_data[['temp', 'Is_Working_Day', 'Predicted Consumption']].T,
                     labels=dict(x="Date", y="Factor", color="Value"),
                     x=heatmap_data['ds'].dt.date,
                     y=['Temperature', 'Is Working Day', 'Predicted Consumption'],
                     title="Factor Heatmap and Predicted Consumption")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2)

    fig3 = px.scatter(user_data, x='ds', y='temp', color='Is_Working_Day',
                      labels={'ds': 'Date', 'temp': 'Temperature (°C)', 'Is_Working_Day': 'Working Day'},
                      title='Temperature and Working Day Distribution',
                      color_discrete_map={0: 'red', 1: 'green'})
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3)

    st.subheader('📥 Download Forecast')
    csv = forecast_display.to_csv(index=False)
    st.download_button(
        label="Download forecast as CSV",
        data=csv,
        file_name="energy_forecast.csv",
        mime="text/csv",
    )

def qa_bot_page():
    st.title("⚡ Energy Q&A Assistant")
    st.markdown("Ask questions about energy consumption and get informed answers!")

    index, docstore, index_to_docstore_id, embeddings = load_components()
    vector_store = create_vector_store(embeddings, index, docstore, index_to_docstore_id)
    llm = init_language_model()
    prompt = create_prompt()
    retrieval_chain = create_chains(llm, prompt, vector_store)

    general_responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi! How can I help you?",
        "how are you": "I'm just a bot, but I'm here to help! How can I assist you today?",
        "who are you": "I am a Q&A assistant specializing in energy-related topics, but I can also handle general queries.",
        "how can you assist me": "I can answer questions related to energy consumption, ways to reduce energy usage, tips for saving energy, and more. Feel free to ask anything related to energy!",
        "what can you do": "I can provide information on energy usage, tips on saving energy, and help you understand your energy consumption better.",
        "thank you": "You're welcome! If you have any more questions, feel free to ask.",
        "goodbye": "Goodbye! Have an energy-efficient day!",
    }

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    for i, (question, answer) in enumerate(st.session_state.conversation):
        display_chat_message("human", question, "🧑‍💼")
        display_chat_message("bot", answer, "🤖")

    user_question = st.chat_input("Ask a question about energy:")
    
    if user_question:
        display_chat_message("human", user_question, "🧑‍💼")

        with st.spinner("Thinking..."):
            response = get_response(user_question, retrieval_chain, general_responses)
            display_chat_message("bot", response, "🤖")

        st.session_state.conversation.append((user_question, response))

    st.sidebar.title("Tips for Questions")
    st.sidebar.info(
        "Here are some example questions you can ask:\n"
        "- How can I reduce my energy consumption?\n"
        "- What are some energy-efficient appliances?\n"
        "- How does weather affect energy usage?\n"
        "- What is the impact of renewable energy on consumption?\n"
        "- Can you explain the concept of peak demand?"
    )

def about_page():
    st.title('🔍 About This App')
    st.write("""
    This app is designed to provide comprehensive insights into energy consumption through two main features:

    1. **Energy Consumption Forecast**
       - Utilizes a Prophet model for time series forecasting
       - Allows users to input custom data for a 5-day forecast
       - Provides interactive visualizations of the forecast
       - Offers downloadable CSV of forecast data

    2. **Energy Q&A Assistant**
       - Powered by advanced natural language processing
       - Answers questions related to energy consumption and efficiency
       - Uses a knowledge base derived from energy-related documents
       - Provides instant, informative responses to user queries

    **Technologies Used**:
    - Streamlit for the web interface
    - Prophet for time series forecasting
    - Plotly for interactive visualizations
    - Langchain and HuggingFace for natural language processing
    - FAISS for efficient similarity search and clustering of dense vectors

    This app aims to empower users with valuable insights into energy usage patterns and provide expert knowledge on energy-related topics. Whether you're looking to forecast future energy consumption or seeking answers to specific energy questions, this app has you covered!
    """)

def display_chat_message(role, content, avatar):
    message_class = "user" if role == "human" else "bot"
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <div class="avatar">
                <span class="avatar-icon">{avatar}</span>
            </div>
            <div class="message">
                <p>{content}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Energy Insights App", page_icon="⚡", layout="wide")
    set_custom_css()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Forecasting", "Q&A Bot", "About"])

    if page == "Forecasting":
        forecasting_page()
    elif page == "Q&A Bot":
        qa_bot_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()