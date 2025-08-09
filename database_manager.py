import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


class DatabaseManager:
    """
    Manages the creation and management of a vector database from documents.
    """
    
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load environment variables
        load_dotenv()
        
        # Set the embedding model
        self.embedding_model_name = embedding_model_name
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=250,
            length_function=len,
            add_start_index=True,
        )
    
    def load_documents(self, data_path="books"):
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
        if not documents:
            print("No documents to split.")
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        
        # Show example chunk
        # if chunks:
        #     document = chunks[0]
        #     print(document.page_content)
        #     print(document.metadata)
        
        return chunks
    
    def save_to_chroma(self, chunks, persist_directory="chroma"):
        """Save document chunks to Chroma database."""
        if not chunks:
            print("No chunks to save.")
            return
        
        # Clear out the database first
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Cleared existing database at {persist_directory}")
        
        # Create a new DB from the documents
        try:
            db = Chroma.from_documents(
                chunks, 
                self.embedding_function, 
                persist_directory=persist_directory
            )
            # Note: db.persist() is deprecated in newer versions of Chroma
            print(f"Saved {len(chunks)} chunks to {persist_directory}.")
            return db
        except Exception as e:
            print(f"Error saving to Chroma: {e}")
            return None
    
    def generate_data_store(self, data_path="books", persist_directory="chroma"):
        """Main method to generate the complete data store."""
        print(f"Using embedding model: {self.embedding_model_name}")
        print(f"Database will be saved to: {persist_directory}")
        
        # Load documents
        documents = self.load_documents(data_path)
        if not documents:
            print("No documents loaded. Exiting.")
            return False
        
        # Split documents into chunks
        chunks = self.split_text(documents)
        if not chunks:
            print("No chunks created. Exiting.")
            return False
        
        # Save to database
        db = self.save_to_chroma(chunks, persist_directory)
        if db is None:
            print("Failed to create database.")
            return False
        
        print("Database generation completed successfully!")
        return True
    
    def get_database(self, persist_directory="chroma"):
        """Get an existing Chroma database instance."""
        if not os.path.exists(persist_directory):
            print(f"Database not found at {persist_directory}. Run generate_data_store() first.")
            return None
        
        try:
            db = Chroma(
                persist_directory=persist_directory, 
                embedding_function=self.embedding_function
            )
            return db
        except Exception as e:
            print(f"Error loading database: {e}")
            return None

# if __name__ == "__main__":
#     # Example usage
#     db_manager = DatabaseManager(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
    
#     # Generate the database
#     success = db_manager.generate_data_store(data_path="books", persist_directory="chroma")
    
#     if success:
#         print("\nDatabase created successfully!")
        
#         # Test loading the database
#         db = db_manager.get_database("chroma")
#         if db:
#             print("Database loaded successfully for querying.")
