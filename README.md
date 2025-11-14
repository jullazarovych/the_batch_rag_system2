# Multimodal RAG System: The Batch News

A Multimodal Retrieval-Augmented Generation (RAG) Search System designed to answer questions based on the news archive "The Batch" by DeepLearning.AI. The system collects data, indexes text and images in the Weaviate vector database, and provides a Flask web interface for interaction.

## Key Features

* **Multimodal Search:** 
  - **Text:** Hybrid search (BM25 + vector similarity, alpha=0.7) using Sentence Transformers (all-MiniLM-L6-v2)
  - **Images:** CLIP ViT-B/32 embeddings with negative prompt filtering (0.6 weight against diagrams, charts, text)
  - **Parallel retrieval:** 15 text chunks + 5 images per query
  
* **Smart Ranking & Generation:** 
  - Gemini 1.5 Flash as reranker and answer generator
  - Analyzes candidates with priority: **Relevance > Recency > Visual Evidence**
  - Processes both text (6000 chars per article) and images (auto-resized to 800x800, RGB conversion)
  - Automatic model selection (scans for available Flash models)
  
* **Full Article Context:** 
  - Indexes chunks (1000 chars, 200 overlap) using LangChain's RecursiveCharacterTextSplitter
  - Retrieves **full articles** from JSON database (ARTICLES_DB) for complete context
  - Separators: `\n\n`, `\n`, `. `, `! `, `? `, ` `, `""`
  
* **Robust Error Handling:**
  - Automatic retry with exponential backoff for Google API quota limits (429 errors: 20s, 40s, 60s)
  - Image preprocessing (RGB conversion, 800x800 thumbnails, timeout: 3s)
  - Graceful fallbacks for missing data
  - Extended Weaviate timeouts (init: 60s, query: 120s, insert: 120s)
  
* **Evaluation System:**
  - LLM-as-a-Judge approach using Gemini
  - Custom metrics: Faithfulness, Answer Relevancy, Context Precision, Ground Truth Similarity
  - 10-second cooldowns between API calls to avoid rate limits
  
* **Web Interface:** 
  - Flask-based UI displaying generated answer, source articles (top 6), and image gallery (top 4)

---

## Architecture and Data Flow

1. **Data Collection (`scripts/data_collection.py`):** 
   * Scrapes N pages from deeplearning.ai/the-batch/
   * Extracts articles using BeautifulSoup (finds `<h1 id="news">`)
   * Processes content elements: paragraphs, lists, divs
   * Splits into chunks using RecursiveCharacterTextSplitter (1000/200)
   * Stores three JSON files:
     - `data/raw/batch_articles.json` (raw scraped data)
     - `data/processed/batch_chunks.json` (individual chunks with metadata)
     - `data/processed/news_articles.json` (full articles for ARTICLES_DB)

2. **Indexing (`scripts/process_embedings.py`):**
   * Connects to local Weaviate (http://localhost:8080, grpc://localhost:50051)
   * Creates two collections:
     - **BatchChunk:** Text chunks vectorized using Hugging Face text2vec (all-MiniLM-L6-v2)
       - Properties: content, issue_id, issue_date, issue_url, issue_title, news_title, image_url, image_caption
     - **BatchImage:** Images vectorized using local CLIP ViT-B/32
       - Properties: image_url, news_title, issue_id, issue_url
   * Batch import: 10 objects/batch for text, dynamic batching for images
   * Verifies counts after import

3. **Application (`run.py` + `app/`):** 
   * Launches Flask web server on port 5000

4. **Query Flow (UI -> `app/view.py`):** 
   * User enters query through web interface

5. **Orchestration (`services/orchestrator.py`):**
   * On startup: Loads `news_articles.json` into memory (ARTICLES_DB dict, keyed by title)
   * Calls `retrieval_service` for parallel search:
     - **Text search:** Hybrid search (alpha=0.7) on BatchChunk collection (limit: 15)
     - **Image search:** CLIP-based search on BatchImage collection (limit: 5)
   * Collects unique articles by title into `all_candidates_map`
   * For each unique title, pulls full article content from ARTICLES_DB
   * Sorts candidates by date (newest first using ISO date strings)
   * Sends top 6 candidates to `generation_service`
   * Returns: answer, top 6 sources, top 4 gallery images

6. **Retrieval (`services/retrieval_service.py`):**
   * Initializes on import:
     - Connects to Weaviate (localhost:8080, grpc:50051)
     - Loads CLIP ViT-B/32 model to CUDA/CPU
     - Gets BatchChunk and BatchImage collections
   * **Text Search (hybrid):**
     - Weaviate hybrid query with alpha=0.7 (70% vector, 30% BM25)
     - Returns: content, news_title, issue_date, issue_url, image_url
   * **Image Search:**
     - Generates CLIP text embedding for query
     - Creates negative vector from: "diagram", "chart", "text", "abstract art", "screenshot"
     - Subtracts 0.6 * negative_vector from positive_vector
     - Normalizes final vector (L2 norm)
     - Hybrid query with alpha=0.7
     - Returns: image_url, news_title, issue_url

7. **Generation (`services/generation_service.py`):**
   * On startup:
     - Lists all available Gemini models
     - Selects first Flash model with generateContent support
     - Falls back to "models/gemini-1.5-flash" if none found
   * For each candidate:
     - Downloads images (max 800x800, converts P/RGBA/LA to RGB, timeout: 3s)
     - Truncates article content to first 6000 chars
   * Constructs multimodal prompt:
     - System instructions (priority: Relevance > Recency > Visual Evidence)
     - Article metadata (title, date, source_type)
     - Article content (6000 chars max)
     - Associated images (PIL Image objects)
   * Handles errors:
     - Catches `google_exceptions.ResourceExhausted` (429)
     - Exponential backoff: 20s, 40s, 60s (3 retries max)
   * Returns: Markdown-formatted answer with source citations

8. **Response (`app/templates/base.html`):** 
   * Renders the answer, sources, and image gallery

---

## Installation and Startup

### Step 1: Prerequisites

Before you begin, make sure you have:

* [Python 3.10+](https://www.python.org/downloads/)
* [Docker and Docker Compose](https://www.docker.com/products/docker-desktop/)
* [Git](https://git-scm.com/downloads)

### Step 2: Cloning and Installing Dependencies

```bash
# Clone the repository
git clone https://github.com/jullazarovych/the_batch_rag_system2.git
cd the_batch_rag_system2

# Create and activate the virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The installation includes ~60 packages totaling approximately 2GB, including PyTorch, CLIP, Transformers, and other ML libraries.

### Step 3: API Keys and .env

For generation to work, you need a Gemini API key.

**Get Gemini API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Create a new API key.

**Create .env file:**
1. In the root folder of the project, create a file named `.env`.
2. Add your key:

```ini
GEMINI_API_KEY=AIzaSy...
```

> **Note:** The system will automatically select the best available Gemini Flash model (typically `gemini-1.5-flash`).

**Optional: Hugging Face API**

If you prefer to use Hugging Face for embeddings instead of local models:

1. Get your Hugging Face key: Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a free "Read" token.
2. Add the key to `.env`:

```ini
HUGGINGFACE_APIKEY=hf_...
```

> **Warning:** Hugging Face API can be less stable than local models. The default setup uses local Sentence Transformers and CLIP models.

### Step 4: Start Vector Database (Weaviate)

Launch Docker:

```bash
docker compose up -d
```

Verify Weaviate is running:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

Expected response: `{"status": "healthy"}`

### Step 5: Load Data into Database

**Important: Execute this only when Docker is already running**

**Data Collection (Scraping):**

```bash
# This will download ~4 pages of the archive (60 issues), ~150MB
python scripts/data_collection.py
```

> **Note:** You can change `max_pages=4` at the end of `data_collection.py` to:
> - `1` for quick test (~15 issues)
> - `2` for medium test (~30 issues)
> - `6` for full archive (~90 issues, ~200MB)

The script will create:
- `data/raw/batch_articles.json` - Raw scraped data
- `data/processed/batch_chunks.json` - Individual chunks with metadata
- `data/processed/news_articles.json` - Full articles for orchestrator

**Indexing (Vectorization):**

```bash
# This will load data into Weaviate (may take 5-15 minutes depending on data size)
python scripts/process_embedings.py
```

The script will:
1. Connect to Weaviate
2. Delete old schemas (if they exist)
3. Create BatchChunk and BatchImage collections
4. Import text chunks (batch size: 10)
5. Download and vectorize images using CLIP
6. Verify final counts

Expected output example:
```
Text chunk count in Weaviate: 450
Image count in Weaviate: 60
```

### Step 6: Run Web Application

```bash
python run.py
```

Your server is running! Open in your browser: **http://127.0.0.1:5000**

---

## (Optional) Quality Evaluation

The system includes a custom evaluation framework using **LLM-as-a-Judge** approach (alternative to RAGAS).

**Step 1: Prepare Test Data**

Create your test questions and answers manually in `scripts/evaluation_data.py`:

```python
test_questions = [
    "Your question here?",
    # ... more questions
]

ground_truths = [
    ["Expected answer here."],
    # ... more answers
]
```

> **Note:** The repository includes 10 pre-written test questions from recent "The Batch" issues.

**Step 2: Run Evaluation**

```bash
# This will run your custom evaluation script using LLM-as-a-Judge approach
python scripts/own_test_rag.py
```

The script will:
1. Process first 5 questions (configurable in `questions_to_run`)
2. Get RAG system's answer using `orchestrator.get_rag_response()`
3. Wait 10 seconds (cooldown between API calls)
4. Ask Gemini to judge the answer based on 4 metrics:
   - **Faithfulness** (0.0-1.0): No hallucinations vs context
   - **Answer Relevancy** (0.0-1.0): Relevance to question
   - **Context Precision** (0.0-1.0): Quality of retrieved context
   - **Ground Truth Similarity** (0.0-1.0): Similarity to expected answer
5. Wait 10 seconds (cooldown)
6. Save results to `evaluation_results_custom.csv`

> **Warning:** Evaluation makes 2 API calls per question (RAG + Judge), so rate limits may occur. Default cooldowns: 10 seconds between calls.

**Sample Output:**
```
   question  faithfulness  answer_relevancy  context_precision  ground_truth_similarity
0  What is...          0.9               1.0                0.8                     0.7
1  How does...         0.8               0.9                0.9                     0.8
```

---

## Project Structure

```
rag_thebatch/
├── app/
│   ├── templates/
│   │   └── base.html              # Flask template
│   ├── __init__.py
│   └── view.py                    # Flask routes
├── data/
│   ├── processed/
│   │   ├── batch_chunks.json      # Individual chunks (created by data_collection.py)
│   │   └── news_articles.json     # Full articles (loaded into ARTICLES_DB)
│   └── raw/
│       └── batch_articles.json    # Raw scraped data
├── scripts/
│   ├── data_collection.py         # Web scraper (BeautifulSoup + LangChain splitter)
│   ├── evaluation_data.py         # Test questions & ground truths (manual)
│   ├── own_test_rag.py            # LLM-as-a-Judge evaluation script
│   └── process_embedings.py       # Weaviate indexing (CLIP + Sentence Transformers)
├── services/
│   ├── __init__.py
│   ├── generation_service.py      # Gemini answer generation with retry logic
│   ├── orchestrator.py            # RAG orchestration (ARTICLES_DB, search, ranking)
│   └── retrieval_service.py       # Weaviate search (hybrid text + CLIP images)
├── weaviate_data/                 # Docker volume for Weaviate persistence
├── .env                           # API keys (GEMINI_API_KEY, optional HUGGINGFACE_APIKEY)
├── .flaskenv                      # Flask configuration
├── .gitignore
├── docker-compose.yml             # Weaviate configuration
├── evaluation_results_custom.csv  # Evaluation results (created by own_test_rag.py)
├── requirements.txt               # Python dependencies (~60 packages)
└── run.py                         # Flask application entry point
```

---

## Technologies Used

### Core Framework
* **Python 3.10+** - Core programming language
* **Flask 3.1.2** - Web framework
* **Weaviate 4.18.0** - Vector database (HTTP: 8080, gRPC: 50051)

### Machine Learning & Embeddings
* **Sentence Transformers 5.1.2** (all-MiniLM-L6-v2) - Text embeddings via Hugging Face
* **CLIP 1.0.1** (ViT-B/32) - Image embeddings with negative prompt filtering
* **PyTorch 2.9.0** - Deep learning framework for CLIP
* **Torchvision 0.24.0** - Image transformations

### Generation & Orchestration
* **Google Generative AI 0.8.5** (Gemini 1.5 Flash) - Answer generation and LLM-as-a-Judge
* **LangChain Text Splitters 1.0.0** - RecursiveCharacterTextSplitter for chunking
* **LangChain Hugging Face 1.0.1** - Embedding integration

### Data Processing
* **BeautifulSoup4 4.14.2** - Web scraping
* **Requests 2.32.5** - HTTP library for scraping and image downloads
* **Pillow 12.0.0** - Image processing (RGB conversion, thumbnailing)
* **Pandas 2.3.3** - Evaluation results analysis

### Evaluation
* **Datasets 4.4.1** - Dataset handling for evaluation
* **NumPy 1.26.4** - Numerical operations

### Infrastructure
* **Docker & Docker Compose** - Containerization for Weaviate
* **python-dotenv 1.2.1** - Environment variable management

---

## Troubleshooting

### Docker Issues

If Weaviate fails to start:
```bash
docker compose down
docker compose up -d
```

Check Weaviate health:
```bash
curl http://localhost:8080/v1/.well-known/ready
```

Check Docker logs:
```bash
docker compose logs weaviate
```

### Port Already in Use

If port 5000 is already in use:
```bash
# Option 1: Change port in .flaskenv
FLASK_RUN_PORT=5001

# Option 2: Kill process using port 5000
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :5000
kill -9 <PID>
```

### API Key Issues

- Make sure your `.env` file is in the root directory
- Ensure API keys are valid and without quotes
- Check that `.env` is loaded by verifying console output on startup:
  ```
  Gemini model loaded: models/gemini-1.5-flash
  ```

### Gemini API Quota (429 Error)

The system automatically handles quota limits with exponential backoff:
- First retry: 20 seconds
- Second retry: 40 seconds  
- Third retry: 60 seconds

If you consistently hit limits, consider:
- Waiting a few minutes between queries
- Reducing evaluation batch size (`questions_to_run = test_questions[:3]`)
- Increasing cooldown times in `own_test_rag.py` (change `time.sleep(10)` to `time.sleep(30)`)
- Upgrading your Google Cloud quota

### CLIP Model Issues

If CLIP fails to load:
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
```

Check CUDA availability:
```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU available
```

### Weaviate Connection Timeout

If you see timeout errors, the system uses extended timeouts:
- Init: 60 seconds
- Query: 120 seconds
- Insert: 120 seconds

If still timing out:
```bash
# Restart Weaviate
docker compose restart weaviate

# Check memory usage
docker stats weaviate
```

### Missing Images in Results

If images don't display:
- Check that `image_url` fields exist in your data
- Verify images are accessible (not behind authentication)
- System automatically:
  - Converts images to RGB (handles P, RGBA, LA modes)
  - Resizes to 800x800 max
  - Times out after 3 seconds per image
  - Skips failed images gracefully

### Scraping Issues

If `data_collection.py` fails:
```bash
# Check internet connection
ping deeplearning.ai

# Verify site structure hasn't changed
curl https://www.deeplearning.ai/the-batch/ | grep "issue-"

# Reduce max_pages for testing
# In data_collection.py, change:
collector.collect_data(max_pages=1)  # Instead of 4
```

### Indexing Failures

If `process_embedings.py` shows errors:
```bash
# Check Weaviate is running
curl http://localhost:8080/v1/.well-known/ready

# Check data files exist
ls data/processed/

# View failed objects in output
# Script shows first 5 failed objects with details
```

Common indexing issues:
- Empty content fields (automatically skipped)
- Invalid data types (script converts to strings/ints)
- Network timeouts downloading images (automatically retried)

### Evaluation Errors

If `own_test_rag.py` fails:
```bash
# Make sure Weaviate is running
docker compose ps

# Check evaluation_data.py exists
ls scripts/evaluation_data.py

# Verify Gemini API key
cat .env | grep GEMINI

# Run with fewer questions
# In own_test_rag.py, change:
questions_to_run = test_questions[:2]  # Test with 2 questions first
```

---

## Performance Tips

### Speed Optimization
- **Use GPU:** CLIP runs significantly faster on CUDA-enabled GPUs
- **Reduce batch size:** If experiencing memory issues, lower `max_pages` in data_collection.py
- **Cache results:** Weaviate automatically caches frequent queries

### Cost Optimization
- **Local embeddings:** Default setup uses local models (no API costs for embeddings)
- **Gemini limits:** Free tier provides 15 requests/minute, 1500 requests/day
- **Evaluation:** Process questions in small batches to avoid rate limits

### Quality Optimization
- **Adjust alpha:** Change `alpha=0.7` in retrieval_service.py (higher = more vector search, lower = more keyword search)
- **Tune negative prompts:** Modify negative_concepts in `search_images_by_text()` for better image filtering
- **Increase context:** Raise `content_preview = art['content'][:6000]` limit in generation_service.py (warning: higher costs)

---

**Project Link:** [https://github.com/jullazarovych/the_batch_rag_system2](https://github.com/jullazarovych/the_batch_rag_system2)
**VIDEO DEMO:** [DEMO](https://drive.google.com/file/d/1Dhl6S1vP4P-1zsNKpxkcgoOv7xIzjPrB/view?usp=sharing)

---
