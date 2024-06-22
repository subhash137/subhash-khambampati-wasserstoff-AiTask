
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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

retriever = vector_store.as_retriever()
#create llm


repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"max_new_tokens":500})
api_key = "sk-proj-5NgthFxlyrQQpVr6T5zQT3BlbkFJvFx7gcVPj73K6ZiHX5Xu"
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=api_key)
# prompt = f

from langchain_core.prompts import PromptTemplate

ChainOfThought = """
You are a highly capable AI assistant powered by a Retrieval-Augmented Generation (RAG) model. 
This allows you to retrieve relevant information from a large knowledge base and then use that information 
to generate a final answer.

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





"""

prefix = """You are a helpful chatbot and answer questions based on provided context only.
             If the answer to the question is not there in the context, you can politely say 
             that you do not have the answer
"""
suffix = """


context - {context}

User - {input}

System - 

"""

prompt = PromptTemplate.from_examples(
    examples = ChainOfThought, suffix=suffix , input_variables=["context", "input"], prefix=prefix
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)



        
print(chain.invoke({"input": "What is langchain"}))
print(chain.invoke({"input": "Hi hello how are you?"}))





