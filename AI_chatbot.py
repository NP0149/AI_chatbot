import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import os

# Use environment variable for security
OpenAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Set this in your system

st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

if file is not None:
    # Extract text from PDF
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Break text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user query
    user_query = st.text_input("Type your query here")

    if user_query:
        # Semantic search
        matching_chunks = vector_store.similarity_search(user_query)

        if matching_chunks:
            # Define LLM
            llm = ChatOpenAI(
                api_key=OpenAI_API_KEY,
                max_tokens=300,
                temperature=0,
                model="gpt-3.5-turbo"
            )

            # Create a customized prompt
            customized_prompt = ChatPromptTemplate.from_template(
                """You are my assistant tutor. Answer the question based on the following context.
                If you do not have enough information, simply say "I don't know Jenny".

                Context:
                {context}

                Question: {input}"""
            )

            # Create chain
            chain = create_stuff_documents_chain(llm=llm, prompt=customized_prompt)

            # Generate output
            output = chain.invoke({
                "input": user_query,
                "input_documents": matching_chunks
            })

            st.write(output)
        else:
            st.write("No relevant context found for your query.")
