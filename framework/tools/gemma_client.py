import os
import time
import datetime
try:
    from google import genai
except ImportError:
    print("Warning: google-genai library not found. Install with: pip install google-genai")
    genai = None

from dotenv import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

# --- CONFIGURATION ---
INITIATION_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
HISTORY_FILE = os.path.join("knowledge_base", "gemma_history_log.md")
RUNTIME_HISTORY_FILE = os.path.join("toolout", "gemma_responses", INITIATION_TIME, f"gemma_runtime_history_log.md")

RATE_LIMIT_DELAY = 5.0  # Minimum seconds between API calls
_LAST_CALL_TIMESTAMP = 0.0

def _log_interaction(query: str, response: str):
    """
    Appends the Q&A to a markdown file in the knowledge base directory.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(RUNTIME_HISTORY_FILE), exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = (
        f"\n\n## Gemma Query - {timestamp}\n"
        f"**Question:** {query}\n\n"
        f"**Answer:**\n{response}\n"
        f"---\n"
    )
    
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)

        with open(RUNTIME_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)

        print(f"   [System] Saved Gemma response to {HISTORY_FILE}")
    except Exception as e:
        print(f"   [System] Warning: Failed to log to knowledge base: {e}")

def get_runtime_history_file() -> str:
    """
    Returns the contents of the runtime history log file.
    """
    try:
        with open(RUNTIME_HISTORY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No runtime history log found."

def query_gemma_cloud(
        query: str, 
        model_name: str = "gemma-3-27b-it",
        system_message: str = 
        (
            "You are a knowledge provider for an agentic framework which does HEP research. \n"
            "Provide concise, accurate, and relevant answers based on the query. \n"
            "You answers will be part of a library of knowledge for other agents to use so go as deep as you can. \n"
            "You will be answering questions for other LLM agents. \n"
            "Do not ask follow-up questions. \n"
            "Query: "
        )
    ) -> str:
    """
    Queries Google Gemma 3 (27B) with rate limiting and automatic history logging.

    Paid service, use sparingly.    

    Args:
        query: The input query string to send to Gemma.
    Returns:
        The response text from Gemma.
    """
    global _LAST_CALL_TIMESTAMP
    
    api_key = os.getenv("GEMMA_API_KEY")
    if not api_key:
        return "Error: GEMMA_API_KEY environment variable not set."

    if not genai:
        return "Error: google-genai library is missing. Please install it."

    # --- RATE LIMITER ---
    now = time.time()
    elapsed = now - _LAST_CALL_TIMESTAMP
    if elapsed < RATE_LIMIT_DELAY:
        sleep_time = RATE_LIMIT_DELAY - elapsed
        print(f"   [Gemma] Rate limit active. Sleeping for {sleep_time:.2f}s...")
        time.sleep(sleep_time)

    # For now we do not use system message as gemma-3-27b-it does not support it
    query_full = system_message + query

    try:
        client = genai.Client(api_key=api_key)
        
        # Make the call
        response = client.models.generate_content(
            model=model_name,
            contents=query_full,
            config=genai.types.GenerateContentConfig(
                #system_instruction=system_message, # NOT ENABLED FOR THIS MODEL
                #max_output_tokens=3,
                #temperature=0.3,
            ),
        )
        
        # Update Timestamp
        _LAST_CALL_TIMESTAMP = time.time()
        
        result_text = response.text
        
        # --- LOG TO KNOWLEDGE BASE ---
        _log_interaction(query, result_text)
        
        return result_text
        
    except Exception as e:
        raise RuntimeError(f"Gemma API call failed: {e}")
    
if __name__ == "__main__":
    test_query = "Explain the significance of the Higgs boson in particle physics."
    print("Querying Gemma Cloud...")
    answer = query_gemma_cloud(test_query)
    print("Response from Gemma:")
    print(answer)

    # Call this one directly, no need for subprocess
    # It runs test if called as process