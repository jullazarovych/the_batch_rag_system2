from . import retrieval_service
from . import generation_service 

def get_rag_response(user_query):
    context_chunks = []
    
    try:
        context_chunks = retrieval_service.search_text_chunks(user_query, limit=5)
        
        if not context_chunks:
            return "Unfortunately, I did not find any relevant information to answer this question.", []

    except Exception as e:
        print(f"CRITICAL ERROR: Retrieval component failed: {e}")
        return f"A critical error occurred: failed to access the retrieval service. Check Weaviate.", []

    
    try:
        answer, _ = generation_service.generate_answer_from_context(
            user_query, 
            context_chunks
        )
        
        return answer, context_chunks
        
    except Exception as e:
        print(f"WARNING: Generation component failed: {e}. Falling back to retrieval-only.")
        
        fallback_answer = (
            "An error occurred during response generation (Generation service unavailable).\n"
            "Below are the raw search results that were found:"
        )
        return fallback_answer, context_chunks