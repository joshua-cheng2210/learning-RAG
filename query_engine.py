import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # New import
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

class QueryEngine:
    def __init__(self, persist_directory="chroma", 
                 embedding_model_name="sentence-transformers/all-MiniLM-L12-v2",
                 embedding_model_type="huggingface",
                 text_model_name="google/flan-t5-base"):
        """
        Initialize QueryEngine with specified models.
        
        Args:
            persist_directory: Path to the Chroma database
            embedding_model_name: Name of the embedding model
            embedding_model_type: Type of embedding model ("huggingface" or "gemini")
            text_model_name: Name of the text generation model
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.embedding_model_type = embedding_model_type
        self.text_model_name = text_model_name
        
        # Initialize embedding function based on type
        if embedding_model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")  # Changed from GEMINI_API_KEY
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
        
        # Initialize text generation model
        if self.text_model_name.startswith("google/flan"):
            self.hf_pipeline = pipeline(
                "text2text-generation",
                model=self.text_model_name,
                max_length=512,
            )
        elif self.text_model_name.startswith("gpt") or self.text_model_name.startswith("distilgpt"):
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.text_model_name,
                max_length=512,
                do_sample=True,
                pad_token_id=50256
            )
        else:
            self.text_model_name = "google/flan-t5-small"
            self.hf_pipeline = pipeline(
                "text2text-generation",
                model=self.text_model_name,
                max_length=512,
            )
        
        self.model = HuggingFacePipeline(pipeline=self.hf_pipeline)
        
        # Initialize database
        self.db = Chroma(persist_directory=persist_directory, 
                        embedding_function=self.embedding_function)
    
        self.PROMPT_TEMPLATE = """
            Answer the question based only on the following context:

            {context}

            ---

            Answer the question based on the above context: {question}
            here are the options:
            {options}

            Respond only the Letter of the correct options like A, B, C and D. Do not inlcude the source.
            """
        
        print(f"QueryEngine initialized:")
        print(f"  Embedding: {embedding_model_name} ({embedding_model_type})")
        print(f"  Text Generation: {text_model_name}")
        print(f"  Database: {persist_directory}")

    # Rest of your QueryEngine methods remain the same...
    
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
 
    def semantic_search_database(self, query, k=5):
        """Search the database for relevant documents."""
        if self.db is None:
            return []
        
        try:
            results = self.db.similarity_search_with_relevance_scores(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching database: {e}")
            return []
    
    def filter_response(self, response):
        edit_response = response.replace('-', '').strip()
        return edit_response

    def generate_response(self, question, options, context_text):
        """Generate a response using the LLM."""
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

            if result["is_correct"] == False and len(result["response"]) != 1:
                if result["correct_answer"].upper().strip() == "A":
                    alternate_correct_answer = result["options"][0][4:].replace('-', '').strip()
                elif result["correct_answer"].upper().strip() == "B":
                    alternate_correct_answer = result["options"][1][4:].replace('-', '').strip()
                elif result["correct_answer"].upper().strip() == "C":
                    alternate_correct_answer = result["options"][2][4:].replace('-', '').strip()
                elif result["correct_answer"].upper().strip() == "D":
                    alternate_correct_answer = result["options"][3][4:].replace('-', '').strip()
                else:
                    alternate_correct_answer = ""

                if alternate_correct_answer.upper() == result["response"].upper():
                    result["is_correct"] = True
                else:
                    result["is_correct"] = False

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
