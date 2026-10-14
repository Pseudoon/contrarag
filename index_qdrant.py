# save as fix_index.py, run: python fix_index.py
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from dotenv import load_dotenv
import os

load_dotenv()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
qdrant.create_payload_index(
    collection_name="ai_regulations",
    field_name="jurisdiction",
    field_schema=PayloadSchemaType.KEYWORD
)
print("Index created successfully")