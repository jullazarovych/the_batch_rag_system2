from . import retrieval_service
from . import generation_service 

def get_rag_response(user_query):
    context_chunks = []
    combined_images = []
    seen_urls = set() 
    try:
        context_chunks = retrieval_service.search_text_chunks(user_query, limit=5)
        if context_chunks:
            for chunk in context_chunks:
                img_url = chunk.get('image_url')
                if img_url and img_url not in seen_urls:
                    combined_images.append({
                        'image_url': img_url,
                        'news_title': chunk.get('news_title'),
                        'issue_url': chunk.get('issue_url'),
                        'type': 'Context' 
                    })
                    seen_urls.add(img_url)

        clip_images = retrieval_service.search_images_by_text(user_query, limit=3)
        
        if clip_images:
            for img in clip_images:
                img_url = img.get('image_url')
                if img_url and img_url not in seen_urls:
                    combined_images.append({
                        'image_url': img_url,
                        'news_title': img.get('news_title'),
                        'issue_url': img.get('issue_url'),
                        'type': 'CLIP' 
                    })
                    seen_urls.add(img_url)
        
        combined_images = combined_images[:4]

        if not context_chunks:
            return "Unfortunately, I did not find any relevant information to answer this question.", [], combined_images

    except Exception as e:
        print(f"CRITICAL ERROR: Retrieval component failed: {e}")
        return f"A critical error occurred in retrieval service: {e}", [], []

    
    try:
        answer, _ = generation_service.generate_answer_from_context(
            user_query, 
            context_chunks
        )
        
        return answer, context_chunks, combined_images
        
    except Exception as e:
        print(f"WARNING: Generation component failed: {e}. Falling back to retrieval-only.")
        
        fallback_answer = (
            "An error occurred during response generation (Generation service unavailable).\n"
            "Below are the raw search results that were found:"
        )
        return fallback_answer, context_chunks, combined_images