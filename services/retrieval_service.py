import weaviate
from weaviate.connect import ConnectionParams
import torch
import clip
from weaviate.classes.init import AdditionalConfig, Timeout
import numpy as np 

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"(Retriever) CLIP model loaded onto '{device}' for image search.")
except Exception as e:
    print(f"(Retriever) Error loading CLIP model: {e}")
    clip_model = None
try:
    weaviate_client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host="localhost",
            http_port=8080,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
        ),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=60, query=120, insert=120) 
        )
    )
    weaviate_client.connect()
    
    text_collection = weaviate_client.collections.get("BatchChunk")
    image_collection = weaviate_client.collections.get("BatchImage")
    
    print("(Retriever) Weaviate client connected and collections loaded.")

except Exception as e:
    print(f"(Retriever) Error connecting to Weaviate: {e}")
    text_collection = None
    image_collection = None


def search_text_chunks(query: str, limit: int = 5, alpha: float = 0.7) -> list:
    if not text_collection:
        raise ConnectionError("Weaviate 'BatchChunk' collection not available.")
    try:
        response = text_collection.query.hybrid(
            query=query,
            limit=limit,
            alpha=alpha,
            return_properties=[
                "content", "news_title", "issue_date", "issue_url", "image_url"
            ]
        )
        return [chunk.properties for chunk in response.objects]
    except Exception as e:
        print(f"(Retriever) Text search error: {e}")
        return []

def _get_clip_text_vector(text_query: str) -> np.ndarray:
    if not clip_model:
        raise RuntimeError("CLIP model is not loaded.")
        
    with torch.no_grad():
        text_inputs = clip.tokenize([text_query]).to(device)
        text_features = clip_model.encode_text(text_inputs)
        
    return text_features.cpu().numpy().flatten()

def search_images_by_text(query: str, limit: int = 3) -> list:
    if not image_collection:
        raise ConnectionError("Weaviate 'BatchImage' collection not available.")
    if not clip_model:
        raise ConnectionError("CLIP model not available.")

    try:
        positive_vector = _get_clip_text_vector(query)
        negative_concepts = ["diagram", "chart", "text", "abstract art", "screenshot"]
        negative_vector = _get_clip_text_vector(" ".join(negative_concepts))
        
        final_vector = positive_vector - (0.6 * negative_vector)
        
        norm = np.linalg.norm(final_vector)
        final_vector = final_vector / norm
        
        final_vector_list = final_vector.tolist()
        response = image_collection.query.hybrid(
            query=query,
            vector=final_vector_list,
            limit=limit,
            alpha=0.9,
            return_properties=["image_url", "news_title", "issue_url"]
        )
        
        return [img.properties for img in response.objects]

    except Exception as e:
        print(f"(Retriever) Error while searching for images: {e}")
        return []

def close_connection():
    if weaviate_client:
        weaviate_client.close()
        print("(Retriever) Weaviate connection closed.")