import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
import os

HUGGINGFACEHUB_API_TOKEN = "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"
import re

def parse_text(text):
    # Split the text into lines
    lines = text.split('\n')
    
    # Find the line starting with "Question:"
    question_line = next((line for line in lines if line.startswith("Question:")), None)
    
    if question_line:
        # Find the index of the question line
        question_index = lines.index(question_line)
        
        # Get the content after the question
        answer_text = ' '.join(lines[question_index+1:])
        
        # Extract the helpful answer
        helpful_answer_match = re.search(r"Helpful Answer: (.*)", answer_text)
        
        if helpful_answer_match:
            return helpful_answer_match.group(1)
    
    return None
st.title("Web Content Q&A")

# Input field for the user to enter a link
user_link = st.text_input("Enter a website URL:", "https://python.langchain.com/v0.2/docs/introduction/")

if st.button("Load Content"):
    # Load the web content from the user-provided link
    loader = WebBaseLoader(user_link)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': "cpu"})

    # Vectorstore
    vector_store = FAISS.from_documents(text_chunks, embeddings) #vectors 

    # Create llm
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                         repo_id=repo_id, 
                         model_kwargs={"max_new_tokens": 500})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)

    st.session_state['chain'] = chain
    st.success("Content loaded successfully!")

# Rest of your code for conversation_chat, initialize_session_state, and display_chat_history functions

coT = """
You are a highly capable AI assistant powered by a Retrieval-Augmented Generation (RAG) model. This allows you to retrieve relevant information from a large knowledge base and then use that information to generate a final answer.

When given a question, your process is as follows:

1. Analyze the question and identify the key information needed to answer it.
2. Retrieve relevant information from your knowledge base that could help answer the question.
3. Lay out a chain of thought, walking through the reasoning process step-by-step to arrive at a final answer. This includes:
    - Stating the initial question
    - Listing the relevant information retrieved from the knowledge base
    - Explaining how the retrieved information helps answer the question
    - Connecting the dots and drawing logical conclusions
    - Stating the final answer

An example chain of thought could look like:

Question: What is the tallest mountain in Africa?

Retrieved Information:
- Mount Kilimanjaro is a volcanic mountain in Tanzania. It has three volcanic cones: Kibo, Mawenzi, and Shira.
- The highest point on Mount Kilimanjaro is the Uhuru Peak on the Kibo cone, at an elevation of 5,895 meters (19,341 feet).
- Mount Kenya is the second highest mountain in Africa at 5,199 meters (17,057 feet) after Kilimanjaro.

Chain of Thought:
The question asks for the tallest mountain in Africa. From the retrieved information, we learn that Mount Kilimanjaro is a volcanic mountain located in Tanzania, with its highest point being the Uhuru Peak at 5,895 meters. The information also mentions that Mount Kenya is the second highest mountain after Kilimanjaro. Therefore, the tallest mountain in Africa is Mount Kilimanjaro, with an elevation of 5,895 meters at its highest point (Uhuru Peak).

Final Answer: The tallest mountain in Africa is Mount Kilimanjaro, with an elevation of 5,895 meters (19,341 feet) at its highest point (Uhuru Peak).

By walking through the chain of thought, your responses will be more transparent, allowing humans to understand your reasoning process. Please use this approach when answering questions.
Give answer in Four lines

Here is the user's input - 
\n

"""

def conversation_chat(query):
    if 'chain' not in st.session_state:
        return "Please load a website first by entering a URL and clicking 'Load Content'."
    query = coT + query
    result = st.session_state['chain']({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, parse_text(result["answer"])))
    return parse_text(result["answer"])

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Please load a website to start chatting."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Your Query", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()