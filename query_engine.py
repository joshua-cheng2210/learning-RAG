import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline


class QueryEngine:
    """
    Handles querying the vector database and generating responses using LLM.
    """
    
    def __init__(self, persist_directory="chroma", 
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 text_model_name="google/flan-t5-base"):
        """
        Initialize the QueryEngine.
        
        Args:
            persist_directory (str): Path to the Chroma database
            embedding_model_name (str): Name of the HuggingFace embedding model
            text_model_name (str): Name of the HuggingFace text generation model
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.text_model_name = text_model_name
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Initialize text generation pipeline
        self.llm_pipeline = None
        self.model = None
        
        # Default prompt template
        self.PROMPT_TEMPLATE = """
            Answer the question based only on the following context:

            {context}

            ---

            Answer the question based on the above context: {question}
            here are the options:
            {options}

            Respond only the Letter of the correct options like A, B, C and D
            """
        self.initialize_llm()
    
    def load_quiz_data(self, quiz_file_path='test_questions.json'):
        """Load quiz data from JSON file."""
        try:
            with open(quiz_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"Loaded {len(data)} questions from {quiz_file_path}")
                return data
        except FileNotFoundError:
            print(f"Error: {quiz_file_path} file not found!")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []
    
    def initialize_llm(self):
        """Initialize the language model pipeline."""
        if self.llm_pipeline is None:
            print(f"Creating HuggingFace pipeline with model: {self.text_model_name}")
            try:
                self.llm_pipeline = pipeline(
                    "text2text-generation",
                    model=self.text_model_name,
                    max_length=512,
                )
                
                # Wrap the pipeline in LangChain
                self.model = HuggingFacePipeline(pipeline=self.llm_pipeline)
                print("HuggingFace pipeline created successfully.")
                
            except Exception as e:
                print(f"Error creating HuggingFace pipeline: {e}")
                return False
        return True
    
    def get_database(self):
        """Get the Chroma database instance."""
        try:
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_function)
            return db
        except Exception as e:
            print(f"Error loading database from {self.persist_directory}: {e}")
            print("Make sure you've run DatabaseManager.generate_data_store() first.")
            return None
    
    def semantic_search_database(self, query, k=5):
        """Search the database for relevant documents."""
        db = self.get_database()
        if db is None:
            return []
        
        try:
            results = db.similarity_search_with_relevance_scores(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching database: {e}")
            return []
    
    def filter_response(self, response):
        edit_response = response.replace('-', '').strip()
        return edit_response

    def generate_response(self, question, options, context_text):
        """Generate a response using the LLM."""
        if not self.initialize_llm():
            return "Error: Could not initialize language model."
        
        # Format the prompt
        options_text = "\n".join(options) if isinstance(options, list) else str(options)
        prompt = self.PROMPT_TEMPLATE.format(
            context=context_text, 
            question=question, 
            options=options_text
        )
        
        try:
            # Use the HuggingFace model to generate response
            response_text = self.model.invoke(prompt)
            response_text = self.filter_response(response_text)
            return response_text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response."
    
    def query_single_question(self, question, options=None, show_context=False):
        """Query a single question and return the response."""
        # Search the database
        results = self.semantic_search_database(question, k=5)
        
        if not results:
            return {
                'question': question,
                'response': 'No relevant context found.',
                'context': '',
                'sources': []
            }
        
        # Prepare context from search results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
        
        # Generate response
        response_text = self.generate_response(question, options or [], context_text)
        
        result = {
            'question': question,
            'response': response_text.replace('-', '').strip(),
            'sources': sources
        }
        
        if show_context:
            result['context'] = context_text
        
        return result
    
    def run_quiz(self, quiz_file_path='test_questions.json', show_details=True, limit=None):
        """Run the complete quiz and return results."""
        # Load quiz data
        quiz_data = self.load_quiz_data(quiz_file_path)
        
        if not quiz_data:
            print("No quiz data loaded. Exiting.")
            return []
        
        # Limit questions if specified
        if limit:
            quiz_data = quiz_data[:limit]
            print(f"Running quiz with {limit} questions.")
        
        results = []
        correct_count = 0
        
        for i, question_data in enumerate(quiz_data, 1):
            print(f"Question {i} of {len(quiz_data)}")
            
            question_id = question_data.get("id", i)
            question = question_data["question"]
            options = question_data["options"]
            correct_answer = question_data["answer"]
            
            # Query the database and generate response
            result = self.query_single_question(question, options, show_context=False)
            
            # Add quiz-specific information
            result.update({
                'id': question_id,
                'options': options,
                'correct_answer': correct_answer,
                'response' : result['response'],
                'is_correct': result['response'].strip().upper() == correct_answer.upper()
            })
            
            if result['is_correct']:
                correct_count += 1
            
            results.append(result)
            
            # Show details if requested
            if show_details:
                print("=" * 50)
                print(f"Question {question_id}: {question}")
                for j, option in enumerate(options, 1):
                    print(f"  {option}")
                print(f"AI Response: {result['response']}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Result: {'✓ Correct' if result['is_correct'] else '✗ Incorrect'}")
                print()
        
        # Summary
        accuracy = (correct_count / len(quiz_data)) * 100 if quiz_data else 0
        print(f"\nQuiz Summary:")
        print(f"Total Questions: {len(quiz_data)}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        return results
    
    def set_prompt_template(self, new_template):
        """Set a custom prompt template."""
        self.PROMPT_TEMPLATE = new_template
        print("Prompt template updated.")


# if __name__ == "__main__":
#     # Example usage
#     query_engine = QueryEngine(
#         persist_directory="chroma",
#         embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
#         text_model_name="google/flan-t5-base"
#     )
    
#     # Run a small quiz
#     print("Running quiz...")
#     results = query_engine.run_quiz("test_questions.json", limit=5)
    
#     # Query a single question
#     print("\nSingle question example:")
#     result = query_engine.query_single_question(
#         "Who does Alice follow down the rabbit hole?",
#         ["A - The White Rabbit", "B - The Cheshire Cat", "C - The Mad Hatter", "D - The Queen"],
#         show_context=True
#     )
#     print(f"Response: {result['response']}")
