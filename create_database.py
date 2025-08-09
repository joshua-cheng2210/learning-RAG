from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
# openai.api_key = os.environ['OPENAI_API_KEY']
# openAIWords2Vector = OpenAIEmbeddings()

model_options=[
    "sentence-transformers/all-MiniLM-L6-v2", 
    "sentence-transformers/all-mpnet-base-v2", 
    "BAAI/bge-large-en", 
    "intfloat/e5-base-v2", 
    "SproutsAI/embedding-model",
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "sentence-transformers/all-MiniLM-L12-v2"
    ]

model_option_index = 0
HuggingFaceWords2Vector = HuggingFaceEmbeddings(model_name=model_options[model_option_index])

CHROMA_PATH = f"chroma_{model_options[model_option_index].replace('/', '_').replace('-', '_')}"
DATA_PATH = "books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# loads the documents from the books directory
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# splits the documents into chunks of X characters with Y character overlap
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # document = chunks[0]
    # print(document.page_content)
    # print(document.metadata)

    return chunks

# converts the chunks into vector embeddings and then saves them to the chroma database
def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, HuggingFaceWords2Vector, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# if __name__ == "__main__":
#     main()
