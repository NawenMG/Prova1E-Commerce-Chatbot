from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from pymilvus import Collection
from sentence_transformers import SentenceTransformer

class ActionCercaProdottoMilvus(Action):
    def name(self):
        return "action_cerca_prodotto_milvus"

    def __init__(self):
        # Connetti a Milvus e carica il modello di embedding
        self.collection = Collection("ecommerce_embeddings")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        # Ottieni la query dell'utente
        query = tracker.latest_message.get("text")
        if not query:
            dispatcher.utter_message(text="Non ho capito cosa stai cercando.")
            return []

        # Genera l'embedding per la query
        query_embedding = self.model.encode(query).tolist()

        # Cerca i prodotti simili in Milvus
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=5
        )

        # Processa i risultati
        if results[0]:
            risposta = "Ecco i prodotti che ho trovato:\n"
            for result in results[0]:
                metadata = result.entity.get("metadata")
                risposta += f"- {metadata.get('descrizione')} (Categoria: {metadata.get('categoria')})\n"
        else:
            risposta = "Non ho trovato prodotti simili alla tua richiesta."

        # Rispondi all'utente
        dispatcher.utter_message(text=risposta)
        return []
