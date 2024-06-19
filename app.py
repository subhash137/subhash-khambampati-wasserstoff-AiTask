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


loader = WebBaseLoader("https://wordpress.com/blog/")
documents = loader.load()

#split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
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
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)


query = "Give two posts from the wordpress news letter"
chat_history =[]
result = chain({"question": query, "chat_history":chat_history})
print(result["answer"])
