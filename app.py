from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from langchain import hub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub

HUGGINGFACEHUB_API_TOKEN  =  "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"


loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
documents = loader.load()

#split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})


#vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)

#create llm

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"max_new_tokens":500})



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":5}),
                                              memory=memory)

def extract_helpful_answer(text):
    # Identifying the start and end markers for the helpful answer
    start_marker = "Helpful Answer:"
    start_index = text.find(start_marker) + len(start_marker)
    
    # Extracting the helpful answer from the text
    helpful_answer = text[start_index:].strip()
    
    return helpful_answer

def tooluse(query,chat_history):
    prompt = f"  ###chat history - {chat_history} ###question- {query} If You know the answer to this question, Give the answer if you don't know the answer just output NO"
    response = llm(prompt)
    if response == "NO":
        print('Hello')
        result = chain({"question": query, "chat_history":chat_history})
        chat_history.append({'query':query, 'response':extract_helpful_answer(result["answer"])})
        print(extract_helpful_answer(result["answer"]))
    else:
        chat_history.append({'query':query, 'response':response})
        print(response)


query = "What is langchain?"

chat_history =[]
result = chain({"question": query, "chat_history":chat_history})
chat_history.append({'query':query, 'response':extract_helpful_answer(result["answer"])})


print(extract_helpful_answer(result["answer"]))
query = "What is my previous question?"
# result = chain({"question": query, "chat_history":chat_history})



tooluse(query,chat_history)

# chat_history.append({'query':query, 'response':result["answer"]})


# print(extract_helpful_answer(result["answer"]))