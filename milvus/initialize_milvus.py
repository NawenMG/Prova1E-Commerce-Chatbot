from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# Connetti a Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Carica il modello di embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connetti alla collezione
collection = Collection("ecommerce_embeddings")

# Dati da inserire
prodotti = [
    {"text": "Scarpe da corsa per uomini", "categoria": "scarpe"},
    {"text": "Zaini impermeabili", "categoria": "zaini"},
    {"text": "Smartphone di ultima generazione", "categoria": "elettronica"}
]

# Genera e inserisci embedding
for prodotto in prodotti:
    embedding = model.encode(prodotto["text"]).tolist()
    collection.insert([[embedding], [{"categoria": prodotto["categoria"], "descrizione": prodotto["text"]}]])
    print(f"Inserito: {prodotto['text']}")

print("Tutti i prodotti sono stati inseriti.")
