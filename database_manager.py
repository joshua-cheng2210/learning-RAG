from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_experimental.text_splitter import SemanticChunker
import os
from pathlib import Path
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
        
        # Define different chunking strategies for different file formats
        self.chunking_strategies = {
            'markdown': {
                'chunk_size': 800,      # Larger chunks for structured content
                'chunk_overlap': 150,   # More overlap to preserve context
                'splitter_type': 'markdown'
            },
            'txt': {
                'chunk_size': 500,      # Smaller chunks for plain text
                'chunk_overlap': 0,   # Standard overlap
                'splitter_type': 'recursive'
            },
            'default': {
                'chunk_size': 500,
                'chunk_overlap': 0,
                'splitter_type': 'recursive'
            }
        }
        
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
        """Load documents from the specified directory, supporting multiple formats."""
        try:
            all_documents = []
            
            # Load markdown files
            md_loader = DirectoryLoader(data_path, glob="*.md")
            md_documents = md_loader.load()
            all_documents.extend(md_documents)
            if md_documents:
                print(f"Loaded {len(md_documents)} markdown documents")
            
            # Load text files
            txt_loader = DirectoryLoader(data_path, glob="*.txt")
            txt_documents = txt_loader.load()
            all_documents.extend(txt_documents)
            if txt_documents:
                print(f"Loaded {len(txt_documents)} text documents")
            
            print(f"Total documents loaded: {len(all_documents)}")
            return all_documents
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []

    def get_chunking_strategy(self, file_extension):
        """Get chunking strategy based on file format."""
        if file_extension == '.md':
            return self.chunking_strategies['markdown']
        elif file_extension == '.txt':
            return self.chunking_strategies['txt']
        else:
            return self.chunking_strategies['default']

    def create_text_splitter(self, strategy):
        """Create appropriate text splitter based on strategy."""
        if strategy['splitter_type'] == 'markdown':
            # return MarkdownTextSplitter(
            #     chunk_size=strategy['chunk_size'],
            #     chunk_overlap=strategy['chunk_overlap'],
            #     length_function=len,
            #     add_start_index=True,
            # )
            return SemanticChunker(
                self.embedding_function,
                breakpoint_threshold_type="percentile"
            )
        else:  # recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=strategy['chunk_size'],
                chunk_overlap=strategy['chunk_overlap'],
                length_function=len,
                add_start_index=True,
            )

    def split_text(self, documents):
        """Split documents using format-aware chunking strategies."""
        try:
            all_chunks = []
            
            # Group documents by file format
            format_groups = {}
            for doc in documents:
                source_path = doc.metadata.get('source', '')
                file_extension = Path(source_path).suffix.lower()
                
                if file_extension not in format_groups:
                    format_groups[file_extension] = []
                format_groups[file_extension].append(doc)
            
            # Process each format group with appropriate strategy
            for file_extension, docs in format_groups.items():
                print(f"Processing {len(docs)} {file_extension or 'unknown'} documents...")
                
                # Get chunking strategy for this format
                strategy = self.get_chunking_strategy(file_extension)
                
                # Create appropriate splitter
                text_splitter = self.create_text_splitter(strategy)
                
                # Split documents
                chunks = text_splitter.split_documents(docs)
                
                # Add format metadata to chunks
                for chunk in chunks:
                    chunk.metadata['file_format'] = file_extension or 'unknown'
                    chunk.metadata['chunking_strategy'] = strategy['splitter_type']
                    chunk.metadata['chunk_size'] = strategy['chunk_size']
                
                all_chunks.extend(chunks)
                print(f"  Created {len(chunks)} chunks using {strategy['splitter_type']} strategy")
            
            print(f"Total chunks created: {len(all_chunks)}")
            return all_chunks
            
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
        """Complete pipeline: load documents, split text, and save to database with format-aware chunking."""
        print(f"Starting data store generation with format-aware chunking...")
        print(f"Data path: {data_path}")
        print(f"Persist directory: {persist_directory}")
        print(f"Embedding model: {self.embedding_model_name} ({self.embedding_model_type})")
        
        # Display chunking strategies
        print("\nChunking strategies:")
        for format_type, strategy in self.chunking_strategies.items():
            print(f"  {format_type}: {strategy['chunk_size']} chars, {strategy['chunk_overlap']} overlap, {strategy['splitter_type']} splitter")
        
        # Load documents
        documents = self.load_documents(data_path)
        if not documents:
            return False
        
        # Split into chunks using format-aware strategy
        chunks = self.split_text(documents)
        if not chunks:
            return False
        
        # Save to database
        db = self.save_to_chroma(chunks, persist_directory)
        return db is not None
    