import os
import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
gemini_model = None

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    try:
        genai.configure(api_key=api_key)
        all_models = [m.name for m in genai.list_models()]
        text_models = [m for m in all_models if "generateContent" in genai.get_model(m).supported_generation_methods]
        flash_models = [m for m in text_models if "flash" in m]
        
        if flash_models:
            selected_model_name = flash_models[0]
        elif text_models:
            selected_model_name = text_models[0]
        else:
            raise ValueError("No suitable Gemini models found.")

        gemini_model = genai.GenerativeModel(selected_model_name)
        print(f"Gemini model loaded: {selected_model_name}")

    except Exception as e:
        print(f"Error initializing Gemini: {e}")


def _download_image(url):
    try:
        response = requests.get(url, timeout=5) 
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Could not download image for Gemini context: {e}")
        return None


def generate_answer_from_context(query, context_chunks):
    if not gemini_model:
        raise ConnectionError("Gemini model is not initialized. Check API key.")
    
    if not context_chunks:
        return "I couldn't find any relevant articles to answer your question.", []

    try:
        prompt_parts = []
        system_instruction = """
        You are an expert AI news assistant for 'The Batch'.
        You have access to both TEXT snippets and IMAGES from the articles.
        
        INSTRUCTIONS:
        1. Answer the user's question based ONLY on the provided context (text and images).
        2. If an image helps explain the answer (like a chart or diagram), refer to it.
        3. Cite specific news stories by their titles.
        4. Use Markdown formatting.
        """
        prompt_parts.append(system_instruction)
        prompt_parts.append(f"USER QUESTION: {query}\n\nCONTEXT SOURCES:")

        sources_titles = []

        for i, chunk in enumerate(context_chunks):
            title = chunk.get('news_title', 'No Title')
            date = chunk.get('issue_date', 'Unknown Date')
            content = chunk.get('content', '')
            img_url = chunk.get('image_url')
            
            text_part = f"\n--- SOURCE {i+1} ---\nTitle: {title}\nDate: {date}\nContent: {content}\n"
            prompt_parts.append(text_part)
            
            if img_url:
                image_data = _download_image(img_url)
                if image_data:
                    prompt_parts.append("Image associated with this article:")
                    prompt_parts.append(image_data) 
            
            if title not in sources_titles:
                sources_titles.append(title)

        prompt_parts.append("\nYOUR ANSWER:")

        response = gemini_model.generate_content(prompt_parts)
        
        return response.text, sources_titles

    except Exception as e:
        print(f"Error generating content: {e}")
        raise e