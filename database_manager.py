from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class DatabaseManager:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L12-v2", 
                 embedding_model_type="huggingface"):
        """
        Initialize DatabaseManager with specified embedding model.
        
        Args:
            embedding_model_name: Name of the embedding model
            embedding_model_type: Type of model ("huggingface" or "gemini")
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model_type = embedding_model_type
        
        # Initialize embedding function based on type
        if embedding_model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models")
            
            # Extract model name (remove 'gemini/' prefix)
            model_name = embedding_model_name.replace("gemini/", "")
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
        elif embedding_model_type == "huggingface":
            self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
        else:  # huggingface
            embedding_model_type == "huggingface"
            self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        print(f"Initialized {embedding_model_type} embedding model: {embedding_model_name}")

    # Rest of your DatabaseManager methods remain the same...
    def load_documents(self, data_path):
        """Load documents from the specified directory."""
        try:
            loader = DirectoryLoader(data_path, glob="*.md")
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from {data_path}")
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []

    def split_text(self, documents):
        """Split documents into chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error splitting text: {e}")
            return []

    def save_to_chroma(self, chunks, persist_directory):
        """Save document chunks to Chroma database."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            db = Chroma.from_documents(
                chunks, 
                self.embedding_function, 
                persist_directory=persist_directory
            )
            print(f"Saved {len(chunks)} chunks to Chroma database at {persist_directory}")
            return db
        except Exception as e:
            print(f"Error saving to Chroma: {e}")
            return None

    def generate_data_store(self, data_path="books", persist_directory="chroma"):
        """Complete pipeline: load documents, split text, and save to database."""
        print(f"Starting data store generation...")
        print(f"Data path: {data_path}")
        print(f"Persist directory: {persist_directory}")
        print(f"Embedding model: {self.embedding_model_name} ({self.embedding_model_type})")
        
        # Load documents
        documents = self.load_documents(data_path)
        if not documents:
            return False
        
        # Split into chunks
        chunks = self.split_text(documents)
        if not chunks:
            return False
        
        # Save to database
        db = self.save_to_chroma(chunks, persist_directory)
        return db is not None