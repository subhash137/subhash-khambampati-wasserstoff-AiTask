# subhash-khambampati-wasserstoff-AiTask

# Web Content Q&A Chatbot

This project is a Streamlit-based web application that allows users to load content from a website and ask questions about it using a conversational AI model.

## Features

- Load web content from a user-provided URL
- Process and embed the content for efficient retrieval
- Engage in a question-answering conversation about the loaded content
- Display chat history in a user-friendly interface

## Technologies Used

- Streamlit: For creating the web application interface
- LangChain: For document loading, text processing, and creating the conversational chain
- HuggingFace: For embeddings and language model

- FAISS: For efficient similarity search and retrieval
- Streamlit Chat: For displaying the chat interface

## Setup and Installation

1. Clone this repository
2. Install the required packages
3. Set up your HuggingFace API token as an environment variable
## Usage

1. Run the Streamlit app:

streamlit run app.py

2. Enter a website URL in the provided input field
3. Click "Load Content" to process the website
4. Start asking questions about the loaded content

## How it Works

1. The app loads content from a user-provided URL
2. The text is split into chunks and embedded using HuggingFace embeddings
3. A vector store is created for efficient retrieval
4. A conversational retrieval chain is set up using a HuggingFace language model
5. Users can ask questions, and the app retrieves relevant information to generate responses

## Note

This project uses the Meta-Llama-3-8B-Instruct model from HuggingFace. Ensure you have the necessary permissions and API access to use this model.

## Future Improvements

- Add support for multiple document types (PDF, DOC, etc.)
- Implement error handling for invalid URLs or connection issues
- Add options to customize the language model and embedding choices
- Improve the parsing of responses to handle various formats

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.

