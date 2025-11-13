from flask import Blueprint, render_template, request
from services import orchestrator, retrieval_service

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET', 'POST'])
def main():
    user_query = ""
    answer = ""
    text_sources = []
    image_results = []

    if request.method == 'POST':
        user_query = request.form.get('query')

        if user_query:
            answer, text_sources = orchestrator.get_rag_response(user_query)
            
            image_results = retrieval_service.search_images_by_text(user_query, limit=3)

    return render_template(
        "base.html", 
        user_query=user_query,
        answer=answer,
        text_sources=text_sources,
        image_results=image_results
    )