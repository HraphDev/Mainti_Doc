import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

load_dotenv()
client = QdrantClient(url=os.environ["QDRANT_URL"])

# (nom, dimension) : BGE-M3 = 1024 ; CLIP multilingue ViT-B-32 = 512
COLLECTIONS = {"maintidoc_text": 1024, "maintidoc_visual": 512}
KEYWORDS = ("document_id", "revision", "equipment_id", "access_groups", "type", "element_id")

for name, dim in COLLECTIONS.items():
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        for f in KEYWORDS:
            client.create_payload_index(name, f, PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "page", PayloadSchemaType.INTEGER)
    print(name, "OK")