# OpenSearch Setting
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200

# Retrieval+Index Setting
EMBEDDING_DIM = 1536
RAG_INDEX_NAME = "tesseract-txt"

# Retrieval Setting
RAG_TOP_K = 5
RRF_RANK_CONSTANT = 60

# Index Setting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_OUTPUT_DIR = "outputs/chunks"
INDEX_FILE_PATTERNS = [
    "rag_opensearch/ocr_tesseract/*.txt",
]

# Gemini Setting
GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-001"
RAG_LLM_MODEL_NAME = "gemini-2.5-pro"
RAG_LLM_TEMPERATURE = 1.0
RAG_LLM_MAX_OUTPUT_TOKENS = 10000
RAG_LLM_THINKING_BUDGET = 128

# OCR Setting
OCR_INPUT_PATTERN = "docs/*.pdf"
OCR_OUTPUT_DIR = "rag_opensearch/ocr_tesseract"
OCR_LANG = "jpn"
OCR_DPI = 300
