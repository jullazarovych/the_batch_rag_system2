import os
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(override=True)

try:
    from services import orchestrator
    from services.generation_service import gemini_model 
    from scripts.evaluation_data import test_questions, ground_truths 
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure Weaviate is running (docker compose up -d)")
    print("Make sure 'google-generativeai' is installed (pip install google-generativeai)")
    sys.exit(1)
except Exception as e:
    print(f"Failed to load modules: {e}")
    sys.exit(1)


def create_judge_prompt(question, generated_answer, context, ground_truth):
    context_str = "\n\n".join(context)
    
    return f"""
    You are an evaluation system (Judge). Your task is to evaluate a RAG system's response based on the provided data.
    
    USER QUESTION:
    "{question}"
    
    GROUND TRUTH (Ideal Answer):
    "{ground_truth}"
    
    RETRIEVED CONTEXT (The information the RAG system found):
    ---
    {context_str[:8000]}... 
    ---
    
    GENERATED ANSWER (The RAG system's actual response):
    ---
    {generated_answer}
    ---
    
    TASKS:
    Please evaluate the GENERATED ANSWER based on the following metrics.
    
    1.  **Faithfulness (0.0 to 1.0):** Does the GENERATED ANSWER contradict or hallucinate information NOT present in the RETRIEVED CONTEXT?
        (0.0 = Major hallucination; 1.0 = Fully faithful to the context).
        
    2.  **Answer Relevancy (0.0 to 1.0):** Is the GENERATED ANSWER relevant to the USER QUESTION?
        (0.0 = Completely irrelevant; 1.0 = Perfectly relevant).
        
    3.  **Context Precision (0.0 to 1.0):** Does the RETRIEVED CONTEXT contain useful information to answer the question, or is it mostly noise?
        (0.0 = All noise; 1.0 = All useful).
        
    4.  **Ground Truth Similarity (0.0 to 1.0):** How similar is the GENERATED ANSWER to the GROUND TRUTH?
        (0.0 = Completely different; 1.0 = Identical meaning).

    OUTPUT FORMAT: Return ONLY a valid JSON object with the scores.
    Example:
    {{
        "faithfulness": 0.9,
        "answer_relevancy": 1.0,
        "context_precision": 0.8,
        "ground_truth_similarity": 0.7
    }}
    """

def run_evaluation():
    if not gemini_model:
        print("Gemini model not loaded in generation_service. Exiting.")
        return

    print("Starting RAG Evaluation (Custom 'Simple Ragas' Mode)...")
    
    all_results = []

    questions_to_run = test_questions[:5]
    truths_to_run = ground_truths[:5]

    for i, question in enumerate(questions_to_run):
        print(f"\n--- Processing Q {i+1}/{len(questions_to_run)}: {question} ---")
        
        answer, text_sources, _ = orchestrator.get_rag_response(question)
        retrieved_contexts = [s['content'] for s in text_sources]
        ground_truth = truths_to_run[i][0]
        
        print(f"RAG Answer: {answer[:100]}...")
        print("...waiting 10 seconds (RAG call cooldown)...")
        time.sleep(10)
        
        judge_prompt = create_judge_prompt(question, answer, retrieved_contexts, ground_truth)
        
        scores = {} 
        
        try:
            print("Asking Judge (Gemini) to score...")
            response = gemini_model.generate_content(
                judge_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            scores_text = response.text.replace("```json", "").replace("```", "") 
            scores = json.loads(scores_text)
            print(f"Scores: {scores}")

        except Exception as e:
            print(f"Error during judging: {e}")
            if "429" in str(e):
                print("Hit Rate Limit (429). Stopping evaluation.")
                break 
        
        all_results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevancy": scores.get("answer_relevancy", 0.0),
            "context_precision": scores.get("context_precision", 0.0),
            "ground_truth_similarity": scores.get("ground_truth_similarity", 0.0)
        })

        print("...waiting 10 seconds (Judge call cooldown)...")
        time.sleep(10)

    df = pd.DataFrame(all_results)
    df.to_csv("evaluation_results_custom.csv", index=False)
    
    print("\n\n=== EVALUATION COMPLETE ===")
    print(df.drop(columns=['answer', 'ground_truth']))
    print("\nDetailed results saved to 'evaluation_results_custom.csv'")

if __name__ == "__main__":
    run_evaluation()