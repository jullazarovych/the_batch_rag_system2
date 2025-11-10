from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Custom text2vec API")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

class Texts(BaseModel):
    text: str

@app.post("/vectors")
def vectorize(payload: Texts):
    vectors = model.encode([payload.text]).tolist()
    return {"vectors": vectors}
