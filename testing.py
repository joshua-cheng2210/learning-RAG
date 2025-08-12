from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables
print(os.getenv("GEMINI_API_KEY"))