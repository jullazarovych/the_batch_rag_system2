import json
import os
import requests
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import torch
import clip

import weaviate
import weaviate.classes.config as wvc
from weaviate.connect import ConnectionParams

try:
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host="localhost",
            http_port=8080,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
        )
    )
    client.connect()

    if not client.is_ready():
        raise Exception("Weaviate is not ready (client.is_ready() returned False)")

    print("Successfully connected to local Weaviate (http://localhost:8080)")

except Exception as e:
    print("ERROR: Could not connect to Weaviate.")
    print("Make sure Docker is running: 'docker compose up -d'")
    print(f"Details: {e}")
    exit()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device '{device}' for CLIP (images)")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

try:
    with open("data/processed/batch_chunks.json", "r", encoding="utf-8") as f:
        batch_chunks = json.load(f)
    with open("data/processed/news_articles.json", "r", encoding="utf-8") as f:
        news_articles = json.load(f)
except FileNotFoundError as e:
    print(f"ERROR: JSON file not found. {e}")
    print("Please run 'python scripts/datacollection.py' first to generate the JSONs.")
    exit()

print(f"Loaded {len(batch_chunks)} chunks and {len(news_articles)} news articles.")

TEXT_CLASS_NAME = "BatchChunk"
IMAGE_CLASS_NAME = "BatchImage"
def create_schemas(client):
    for name in [TEXT_CLASS_NAME, IMAGE_CLASS_NAME]:
        if client.collections.exists(name):
            print(f"Deleting old schema '{name}'...")
            client.collections.delete(name)

    print(f"Creating schema '{TEXT_CLASS_NAME}'...")
    client.collections.create(
        name=TEXT_CLASS_NAME,
        description="A chunk of text from a 'The Batch' news article",
        
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_huggingface(
            model="sentence-transformers/all-MiniLM-L6-v2",
            vectorize_collection_name=False
        ),
        
        properties=[
            wvc.Property(name="content", data_type=wvc.DataType.TEXT),
            wvc.Property(name="issue_id", data_type=wvc.DataType.INT),
            wvc.Property(name="issue_date", data_type=wvc.DataType.TEXT),
            wvc.Property(name="issue_url", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="issue_title", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="news_title", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="image_url", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="image_caption", data_type=wvc.DataType.TEXT, skip_vectorization=True),
        ],
    )

    print(f"Creating schema '{IMAGE_CLASS_NAME}'...")
    client.collections.create(
        name=IMAGE_CLASS_NAME,
        description="An image from a 'The Batch' news article",
        vectorizer_config=wvc.Configure.Vectorizer.none(),
        
        properties=[
            wvc.Property(name="image_url", data_type=wvc.DataType.TEXT),
            wvc.Property(name="news_title", data_type=wvc.DataType.TEXT),
            wvc.Property(name="issue_id", data_type=wvc.DataType.INT),
            wvc.Property(name="issue_url", data_type=wvc.DataType.TEXT),
        ],
    )

    print("Schemas created successfully!")


def get_image_embedding(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(img_preprocessed)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

def import_text_data(client, chunks_data):
    print(f"\n--- Importing {len(chunks_data)} text chunks ---")
    text_collection = client.collections.get(TEXT_CLASS_NAME)

    failed_objects = []

    with text_collection.batch.dynamic() as batch:
        for chunk in tqdm(chunks_data):
            content = chunk.get("content")
            if not content or not content.strip():
                continue

            try:
                props = {
                    "content": str(content),
                    "issue_id": int(chunk.get("issue_id") or 0),
                    "issue_date": str(chunk.get("issue_date") or ""),
                    "issue_url": str(chunk.get("issue_url") or ""),
                    "issue_title": str(chunk.get("issue_title") or ""),
                    "news_title": str(chunk.get("news_title") or ""),
                    "image_url": str(chunk.get("image", {}).get("url") or ""),
                    "image_caption": str(chunk.get("image", {}).get("caption") or ""),
                }

                batch.add_object(properties=props)

            except Exception as e:
                failed_objects.append({"object": chunk, "error": str(e)})

    if text_collection.batch.failed_objects or failed_objects:
        total_failed = len(text_collection.batch.failed_objects) + len(failed_objects)
        print(f"Error: {total_failed} objects failed to import.")
        for i, f in enumerate(text_collection.batch.failed_objects[:5]):
            print(f"\n--- Failed object {i} from Weaviate Batch ---")
            print(f"Message: {f.message}") 
            
            if f.object_: 
                print(f"Properties: {f.object_.properties}") 
            else:
                print("Properties: (object data not available)")

        for i, f in enumerate(failed_objects[:5]):
            print(f"\n--- Failed object {i} from local catch ---")
            print(f"Object: {f['object']}")
            print(f"Error message: {f['error']}")
    else:
        print("Text chunks imported successfully.")


def import_image_data(client, articles_data):
    print(f"\n--- Importing images from {len(articles_data)} articles ---")
    image_collection = client.collections.get(IMAGE_CLASS_NAME)
    processed = 0

    with image_collection.batch.dynamic() as batch:
        for article in tqdm(articles_data):
            image_info = article.get("image")
            if not image_info or not image_info.get("url"):
                continue

            vector = get_image_embedding(image_info["url"])
            if vector is None:
                continue

            props = {
                "image_url": image_info["url"],
                "news_title": article.get("title", ""),
                "issue_id": article.get("issue_id", 0),
                "issue_url": article.get("issue_url", ""),
            }
            batch.add_object(properties=props, vector=vector.tolist())
            processed += 1

    print(f"Imported {processed} images successfully.")


if __name__ == "__main__":
    create_schemas(client)
    import_text_data(client, batch_chunks)
    import_image_data(client, news_articles)

    print("\n=== Verification ===")
    text_collection = client.collections.get(TEXT_CLASS_NAME)
    image_collection = client.collections.get(IMAGE_CLASS_NAME)

    text_count = text_collection.aggregate.over_all(total_count=True).total_count
    image_count = image_collection.aggregate.over_all(total_count=True).total_count

    print(f"Text chunk count in Weaviate: {text_count}")
    print(f"Image count in Weaviate: {image_count}")

    client.close()
    print("Connection to Weaviate closed.")
