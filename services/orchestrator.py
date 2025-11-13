from . import retrieval_service
from . import generation_service 

def get_rag_response(user_query):
    unique_articles = []
    seen_titles = set()
    shown_image_urls = set() 
    clip_images = [] 
    
    try:
        raw_chunks = retrieval_service.search_text_chunks(user_query, limit=50)
        
        if raw_chunks:
            for chunk in raw_chunks:
                title = chunk.get('news_title')
                
                if title and title not in seen_titles:
                    unique_articles.append(chunk)
                    seen_titles.add(title)
                    
                    if chunk.get('image_url'):
                        shown_image_urls.add(chunk.get('image_url'))
                
                if len(unique_articles) >= 5:
                    break
        
        if not unique_articles:
            clip_images = retrieval_service.search_images_by_text(user_query, limit=3)
            return "Unfortunately, I did not find any relevant information.", [], clip_images

        raw_clip_images = retrieval_service.search_images_by_text(user_query, limit=5)
        
        if raw_clip_images:
            for img in raw_clip_images:
                url = img.get('image_url')
                if url and url not in shown_image_urls:
                    img['type'] = 'CLIP' 
                    clip_images.append(img)
                    shown_image_urls.add(url) 
        
        clip_images = clip_images[:3]

    except Exception as e:
        print(f"CRITICAL ERROR: Retrieval component failed: {e}")
        return f"A critical error occurred: {e}", [], []

    try:
        answer, _ = generation_service.generate_answer_from_context(
            user_query, 
            unique_articles 
        )
        return answer, unique_articles, clip_images
        
    except Exception as e:
        print(f"WARNING: Generation component failed: {e}. Falling back.")
        
        fallback_answer = (
            "An error occurred during response generation.\n"
            "Below are the top unique articles found:"
        )
        return fallback_answer, unique_articles, clip_images