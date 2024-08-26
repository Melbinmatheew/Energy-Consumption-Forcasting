# import faiss
# import pickle
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.llms import HuggingFaceHub
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.memory import ConversationBufferMemory
# import streamlit as st

# def run_energy_qa_app():
#     # Function to load saved components
#     def load_components():
#         index = faiss.read_index(r"..\Streamlit_app\vector_store.index")
        
#         with open("..\Streamlit_app\docstore.pkl", "rb") as f:
#             docstore = pickle.load(f)
        
#         with open("..\Streamlit_app\index_to_docstore_id.pkl", "rb") as f:
#             index_to_docstore_id = pickle.load(f)
        
#         with open("..\Streamlit_app\embedding.pkl", "rb") as f:
#             embeddings = pickle.load(f)
        
#         return index, docstore, index_to_docstore_id, embeddings

#     # Function to create FAISS vector store
#     def create_vector_store(embeddings, index, docstore, index_to_docstore_id):
#         return FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)

#     # Function to initialize language model
#     def init_language_model():
#         return HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')

#     # Function to create prompt template
#     def create_prompt():
#         prompt_template = """
#         You are a Q&A assistant specializing in energy-related topics based on the content of a provided PDF.
#         Your goal is to provide accurate and concise answers to questions specifically about energy consumption or related subjects covered in the PDF.

#         1. **Answer the question** with a brief paragraph summarizing the relevant information from the PDF. Don't say you are answering from PDF but directly answer the question.
#         2. **Explain the reason for your answer** in a second paragraph by referring to the specific content or details from the PDF that support your response.
#         3. **If the question does not pertain to energy or the content of the PDF**, respond with: "This question is not related to the topics covered in the provided PDF."

#         {context}

#         Question: {input}
#         Answer:
#         """
#         return ChatPromptTemplate.from_template(prompt_template)

#     # Function to create retrieval chain
#     def create_chains(llm, prompt, vector_store):
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = vector_store.as_retriever()
#         return create_retrieval_chain(retriever, document_chain)

#     # Function to get response
#     def get_response(query, retrieval_chain, general_responses):
#         normalized_query = query.lower().strip()
        
#         if normalized_query in general_responses:
#             return general_responses[normalized_query]
#         else:
#             response = retrieval_chain.invoke({"input": query})
#             answer = response["answer"]
            
#             answer_marker = "Answer:"
#             start_index = answer.find(answer_marker)
            
#             if start_index != -1:
#                 generated_output = answer[start_index + len(answer_marker):].strip()
#                 return "\n".join(line.strip() for line in generated_output.splitlines() if line.strip())
#             else:
#                 return "Answer marker not found. Here is the raw response:\n" + answer.strip()

#     # Function to set custom style for the chat messages
#     def set_custom_style():
#         st.markdown("""
#         <style>
#         .chat-message {
#             padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
#         }
#         .chat-message.user {
#             background-color: #2b313e;
#             color: #ffffff;
#             border-bottom-right-radius: 0;
#             margin-left: 40%;
#         }
#         .chat-message.bot {
#             background-color: #475063;
#             color: #ffffff;
#             border-bottom-left-radius: 0;
#             margin-right: 40%;
#         }
#         .chat-message .avatar {
#           width: 20%;
#         }
#         .chat-message .message {
#           width: 80%;
#           padding: 0 1.5rem;
#         }
#         .avatar-icon {
#           font-size: 2rem;
#         }
#         </style>
#         """, unsafe_allow_html=True)

#     # Function to display chat messages in a styled format
#     def display_chat_message(role, content, avatar):
#         message_class = "user" if role == "human" else "bot"
#         with st.container():
#             st.markdown(f"""
#             <div class="chat-message {message_class}">
#                 <div class="avatar">
#                     <span class="avatar-icon">{avatar}</span>
#                 </div>
#                 <div class="message">
#                     <p>{content}</p>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#     # Function to create sidebar with example questions
#     def create_sidebar():
#         st.sidebar.title("Example Questions")
#         st.sidebar.write("""
#         - What are some ways to reduce energy consumption at home?
#         - How can I make my home more energy-efficient?
#         - What is the impact of energy-saving appliances on my electricity bill?
#         - How can I track my energy usage effectively?
#         - What are the benefits of renewable energy sources?
#         """)

#     # Main function to run the Streamlit app
#     def main():
#         # Set page config at the very start
#         # st.set_page_config(page_title="Energy Q&A Assistant", page_icon="⚡", layout="wide")
        
#         set_custom_style()
#         create_sidebar()

#         st.title("✨ Energy Q&A Assistant")
#         st.markdown("Ask questions about energy consumption and get informed answers!")

#         # Load components
#         index, docstore, index_to_docstore_id, embeddings = load_components()
        
#         # Create vector store
#         vector_store = create_vector_store(embeddings, index, docstore, index_to_docstore_id)
        
#         # Initialize language model
#         llm = init_language_model()
        
#         # Create prompt
#         prompt = create_prompt()
        
#         # Create retrieval chain
#         retrieval_chain = create_chains(llm, prompt, vector_store)
        
#         # Define general responses
#         general_responses = {
#             "hello": "Hello! How can I assist you today?",
#             "hi": "Hi! How can I help you?",
#             "how are you": "I'm just a bot, but I'm here to help! How can I assist you today?",
#             "who are you": "I am a Q&A assistant specializing in energy-related topics, but I can also handle general queries.",
#             "how can you assist me": "I can answer questions related to energy consumption, ways to reduce energy usage, tips for saving energy, and more. Feel free to ask anything related to energy!",
#             "what can you do": "I can provide information on energy usage, tips on saving energy, and help you understand your energy consumption better.",
#             "thank you": "You're welcome! If you have any more questions, feel free to ask.",
#             "goodbye": "Goodbye! Have an energy-efficient day!",
#         }
        
#         # Initialize session state for conversation history
#         if 'conversation' not in st.session_state:
#             st.session_state.conversation = []

#         for i, (question, answer) in enumerate(st.session_state.conversation):
#             display_chat_message("human", question, "🫅🏻")
#             display_chat_message("bot", answer, "🧙🏻‍♂️")

#         # Create a text input for user questions
#         user_question = st.chat_input("Ask a question about energy:")
        
#         if user_question:
#             display_chat_message("human", user_question, "🫅🏻")

#             with st.spinner("Thinking..."):
#                 response = get_response(user_question, retrieval_chain, general_responses)
#                 display_chat_message("bot", response, "🧙🏻‍♂️")

#             # Add the new question and answer to the conversation history
#             st.session_state.conversation.append((user_question, response))

#     main()

# # if __name__ == "__main__":
# #     run_energy_qa_app()


import faiss
import pickle
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Constants
VECTOR_STORE_PATH = r"..\Streamlit_app\vector_store.index"
DOCSTORE_PATH = r"..\Streamlit_app\docstore.pkl"
INDEX_TO_DOCSTORE_ID_PATH = r"..\Streamlit_app\index_to_docstore_id.pkl"
EMBEDDING_PATH = r"..\Streamlit_app\embedding.pkl"
HUGGINGFACE_API_TOKEN = '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'

def load_components():
    try:
        # Load FAISS index
        index = faiss.read_index(VECTOR_STORE_PATH)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None, None, None, None
    
    try:
        # Load document store
        with open(DOCSTORE_PATH, "rb") as f:
            docstore = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading document store: {e}")
        return None, None, None, None

    try:
        # Load index-to-docstore ID mapping
        with open(INDEX_TO_DOCSTORE_ID_PATH, "rb") as f:
            index_to_docstore_id = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading index-to-docstore ID mapping: {e}")
        return None, None, None, None

    try:
        # Load embeddings
        with open(EMBEDDING_PATH, "rb") as f:
            embeddings = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, None, None, None

    return index, docstore, index_to_docstore_id, embeddings

def create_vector_store(embeddings, index, docstore, index_to_docstore_id):
    return FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)

def init_language_model():
    return HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token=HUGGINGFACE_API_TOKEN)

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

def set_custom_style():
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
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

def create_sidebar():
    st.sidebar.title("Example Questions")
    st.sidebar.write("""
    - What are some ways to reduce energy consumption at home?
    - How can I make my home more energy-efficient?
    - What is the impact of energy-saving appliances on my electricity bill?
    - How can I track my energy usage effectively?
    - What are the benefits of renewable energy sources?
    """)

def run_energy_qa_app():
    # UI Functions
    set_custom_style()
    create_sidebar()

    st.title("✨ Energy Q&A Assistant")
    st.markdown("Ask questions about energy consumption and get informed answers!")

    # Load components
    index, docstore, index_to_docstore_id, embeddings = load_components()
    if None in [index, docstore, index_to_docstore_id, embeddings]:
        st.stop()

    # Create vector store
    vector_store = create_vector_store(embeddings, index, docstore, index_to_docstore_id)
    
    # Initialize language model
    llm = init_language_model()
    
    # Create prompt
    prompt = create_prompt()
    
    # Create retrieval chain
    retrieval_chain = create_chains(llm, prompt, vector_store)
    
    # Define general responses
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
    
    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    for i, (question, answer) in enumerate(st.session_state.conversation):
        display_chat_message("human", question, "🫅🏻")
        display_chat_message("bot", answer, "🧙🏻‍♂️")

    # Create a text input for user questions
    user_question = st.chat_input("Ask a question about energy:")
    
    if user_question:
        display_chat_message("human", user_question, "🫅🏻")

        with st.spinner("Thinking..."):
            response = get_response(user_question, retrieval_chain, general_responses)
            display_chat_message("bot", response, "🧙🏻‍♂️")

        # Add the new question and answer to the conversation history
        st.session_state.conversation.append((user_question, response))

# If this script is run directly, execute the app
if __name__ == "__main__":
    run_energy_qa_app()
