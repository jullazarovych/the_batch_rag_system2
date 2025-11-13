import os
import time
import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
gemini_model = None

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    try:
        genai.configure(api_key=api_key)
        all_models = [m.name for m in genai.list_models()]
        
        flash_models = [m for m in all_models if "flash" in m and "generateContent" in genai.get_model(m).supported_generation_methods]
        
        if flash_models:
            model_name = flash_models[0]
        else:
            model_name = "models/gemini-1.5-flash" 

        gemini_model = genai.GenerativeModel(model_name)
        print(f"Gemini model loaded: {model_name}")

    except Exception as e:
        print(f"Error initializing Gemini: {e}")

def _download_image(url):
    if not url: return None
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None

def generate_answer_with_ranking(query, candidates):
    if not gemini_model:
        raise ConnectionError("Gemini model is not initialized.")

    if not candidates:
        return "I couldn't find any relevant articles.", []

    prompt_parts = []
    
    system_prompt = f"""
    You are an intelligent news analyst for 'The Batch'.
    USER QUERY: "{query}"
    
    YOUR TASK:
    1. Analyze the provided "CANDIDATE ARTICLES" (text and images).
    2. Select the most relevant information.
    3. **PRIORITY**: Relevance > Recency > Visual Evidence.
    4. Synthesize an answer based *only* on the provided sources.
    5. Cite sources by title. Use Markdown.
    """
    prompt_parts.append(system_prompt)
    prompt_parts.append("\n=== CANDIDATE ARTICLES START ===\n")

    for i, art in enumerate(candidates):
        date_str = str(art.get('date', 'Unknown'))[:10]
        content_preview = art['content'][:6000]
        
        article_text = f"""
        --- ARTICLE {i+1} ---
        Title: {art['title']}
        Date: {date_str}
        Found via: {art.get('source_type', 'text search')}
        Content: {content_preview}
        """
        prompt_parts.append(article_text)
        
        if art.get('image_url'):
            img_obj = _download_image(art['image_url'])
            if img_obj:
                prompt_parts.append(f"Image belonging to Article {i+1}:")
                prompt_parts.append(img_obj)
    
    prompt_parts.append("\n=== CANDIDATE ARTICLES END ===\n")
    prompt_parts.append("\nYOUR ANALYSIS AND ANSWER (in Markdown):")

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt_parts)
            return response.text, [] 

        except google_exceptions.ResourceExhausted:
            wait_time = 20 * (attempt + 1) 
            print(f"Quota exceeded (429). Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            continue 
            
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error: {e}", []

    return "System is currently overloaded (Google API Quota exceeded). Please try again in a few minutes.", []