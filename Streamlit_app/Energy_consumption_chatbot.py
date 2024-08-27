import faiss
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

def run_energy_qa_app():
    # Function to load saved components
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store.index')
    DOCSTORE_PATH = os.path.join(BASE_DIR, 'docstore.pkl')
    EMBEDDING_PATH = os.path.join(BASE_DIR, 'embedding.pkl')

    def load_components():
        try:
            if not os.path.exists(VECTOR_STORE_PATH):
                raise FileNotFoundError(f"File not found: {VECTOR_STORE_PATH}")
            index = faiss.read_index(VECTOR_STORE_PATH)
            
            with open(DOCSTORE_PATH, "rb") as f:
                docstore = pickle.load(f)
            
            # Change this part
            with open(DOCSTORE_PATH, "rb") as f:
                index_to_docstore_id = dict(pickle.load(f))  # Ensure it's a dictionary
            
            with open(EMBEDDING_PATH, "rb") as f:
                embeddings = pickle.load(f)
            
            # Debug: Print information about loaded components
            print(f"Index shape: {index.d}, {index.ntotal}")
            print(f"Docstore size: {len(docstore)}")
            print(f"Index to docstore id size: {len(index_to_docstore_id)}")
            print(f"Index to docstore id type: {type(index_to_docstore_id)}")
            print(f"Embeddings type: {type(embeddings)}")
            
            return index, docstore, index_to_docstore_id, embeddings
        except Exception as e:
            st.error(f"Error loading components: {str(e)}")
            raise

    # Function to create FAISS vector store
    def create_vector_store(embeddings, index, docstore, index_to_docstore_id):
        try:
            # Ensure index_to_docstore_id is a dictionary
            if not isinstance(index_to_docstore_id, dict):
                index_to_docstore_id = dict(index_to_docstore_id)
            
            vector_store = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)
            
            # Test the vector store
            test_query = "energy consumption"
            test_result = vector_store.similarity_search(test_query, k=1)
            print(f"Test query result: {test_result}")
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
    # Function to initialize language model
    def init_language_model():
        return HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')

    # Function to create prompt template
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

    # Function to create retrieval chain
    def create_chains(llm, prompt, vector_store):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        return create_retrieval_chain(retriever, document_chain)

    # Function to get response
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

    # Function to set custom style for the chat messages
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

    # Function to display chat messages in a styled format
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

    # Function to create sidebar with example questions
    def create_sidebar():
        st.sidebar.title("Example Questions")
        st.sidebar.write("""
        - What are some ways to reduce energy consumption at home?
        - How can I make my home more energy-efficient?
        - What is the impact of energy-saving appliances on my electricity bill?
        - How can I track my energy usage effectively?
        - What are the benefits of renewable energy sources?
        """)

    # Main function to run the Streamlit app
    def main():
        set_custom_style()
        create_sidebar()

        st.title("✨ Energy Q&A Assistant")
        st.markdown("Ask questions about energy consumption and get informed answers!")

        try:
            # Load components
            index, docstore, index_to_docstore_id, embeddings = load_components()
            
            # Create vector store
            vector_store = create_vector_store(embeddings, index, docstore, index_to_docstore_id)
            
            if vector_store is None:
                st.error("Failed to create vector store. Please try again later.")
                return

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

            main()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()


