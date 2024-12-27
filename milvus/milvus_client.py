from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

class MilvusClient:
    def __init__(self):
        # Connetti a Milvus e carica il modello di embedding
        connections.connect("default", host="127.0.0.1", port="19530")
        self.collection = Collection("ecommerce_embeddings")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query):
        # Genera embedding della query
        query_embedding = self.model.encode(query).tolist()

        # Cerca in Milvus
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=5
        )

        # Processa i risultati
        products = []
        for result in results[0]:
            metadata = result.entity.get("metadata")
            products.append({"descrizione": metadata.get("descrizione"), "categoria": metadata.get("categoria")})
        return products
