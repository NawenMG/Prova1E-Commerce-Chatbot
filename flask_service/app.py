from concurrent import futures
import grpc
import proto.chatbot_pb2 as chatbot_pb2
import proto.chatbot_pb2_grpc as chatbot_pb2_grpc
from rasa import RasaClient
from milvus.milvus_client import MilvusClient
from pymongo import MongoClient


class ChatbotService(chatbot_pb2_grpc.ChatbotServiceServicer):
    def __init__(self):
        # Configurazione per Rasa e Milvus
        self.rasa_client = RasaClient("http://localhost:5005")  # URL del server Rasa
        self.milvus_client = MilvusClient()

        # Configurazione per MongoDB
        self.mongo_client = MongoClient("mongodb://localhost:27017/")  # Connessione a MongoDB
        self.db = self.mongo_client["chatbot_db"]  # Nome del database
        self.collection = self.db["product_responses"]  # Nome della collezione

    def Query(self, request, context):
        try:
            # Ottieni il messaggio dell'utente
            user_message = request.message

            # Ottieni l'intento dal messaggio tramite Rasa
            intent = self.rasa_client.get_intent(user_message)

            if intent == "cerca_prodotto":
                # Cerca prodotti in Milvus
                results = self.milvus_client.search(user_message)

                # Formatta i risultati come stringa
                response_message = "\n".join(
                    [f"- {r['descrizione']} (Categoria: {r['categoria']})" for r in results]
                )

                # Salva i risultati in MongoDB
                self.collection.insert_one({
                    "user_message": user_message,
                    "intent": intent,
                    "response": results,  # Salva i risultati grezzi come JSON
                })

            else:
                # Risposta per intenti diversi da "cerca_prodotto"
                response_message = "Intento gestito senza Milvus."

            # Costruisci e restituisci la risposta
            return chatbot_pb2.BotResponse(intent=intent, response=response_message)

        except Exception as e:
            # Gestione degli errori
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return chatbot_pb2.BotResponse(intent="error", response="Si è verificato un errore.")


# Configura il server gRPC
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_pb2_grpc.add_ChatbotServiceServicer_to_server(ChatbotService(), server)
    server.add_insecure_port("[::]:50051")
    print("Server gRPC in esecuzione su porta 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
