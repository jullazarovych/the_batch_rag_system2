from . import retrieval_service
from . import generation_service 
import json
import os

try:
    with open("data/processed/news_articles.json", "r", encoding="utf-8") as f:
        articles_data = json.load(f)
    
    ARTICLES_DB = {}
    for art in articles_data:
        full_text = "\n\n".join(art.get('chunks', []))
        
        date_val = art.get('issue_date') or art.get('date') or '1970-01-01'
        
        ARTICLES_DB[art.get('title')] = {
            'content': full_text,
            'date': date_val,
            'url': art.get('issue_url') or art.get('url')
        }
        
    print(f"(Orchestrator) Loaded DB for {len(ARTICLES_DB)} articles.")
except Exception as e:
    print(f"(Orchestrator) Error loading JSON: {e}")
    ARTICLES_DB = {}

def get_rag_response(user_query):
    all_candidates_map = {}
    gallery_images = []

    try:
        text_results = retrieval_service.search_text_chunks(user_query, limit=15)
        
        for chunk in text_results:
            title = chunk.get('news_title')
            
            if title and title not in all_candidates_map:
                db_entry = ARTICLES_DB.get(title)
                
                if db_entry:
                    content = db_entry['content']
                    date = db_entry['date']
                else:
                    content = chunk.get('content')
                    date = chunk.get('issue_date', '1970-01-01')

                if content:
                    article_obj = {
                        'title': title,
                        'date': date, 
                        'content': content,
                        'url': chunk.get('issue_url'),
                        'image_url': chunk.get('image_url'),
                        'source_type': 'text_match'
                    }
                    all_candidates_map[title] = article_obj

        image_results = retrieval_service.search_images_by_text(user_query, limit=5)
        
        for img in image_results:
            title = img.get('news_title')
            
            if img.get('image_url'):
                gallery_obj = {
                    'title': title,
                    'url': img.get('issue_url'),
                    'image_url': img.get('image_url'),
                    'type': 'CLIP'
                }
                gallery_images.append(gallery_obj)

            if title:
                db_entry = ARTICLES_DB.get(title)
                
                if db_entry and title not in all_candidates_map:
                    article_obj = {
                        'title': title,
                        'date': db_entry['date'], 
                        'content': db_entry['content'],
                        'url': img.get('issue_url'),
                        'image_url': img.get('image_url'),
                        'source_type': 'image_match' 
                    }
                    all_candidates_map[title] = article_obj
                    print(f"INFO: Added '{title}' via Image Search (Date: {db_entry['date']})")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return f"Error: {e}", [], []

    final_list = list(all_candidates_map.values())
    
    if not final_list:
        return "No articles found.", [], []

    final_list.sort(key=lambda x: str(x.get('date', ''))[:10], reverse=True)
    
    top_candidates = final_list[:6]

    try:
        answer, _ = generation_service.generate_answer_with_ranking(
            user_query, 
            top_candidates
        )
        return answer, top_candidates, gallery_images[:4]
        
    except Exception as e:
        return f"Gen Error: {e}", top_candidates, gallery_images[:4]