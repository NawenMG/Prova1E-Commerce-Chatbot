import requests

class RasaClient:
    def __init__(self, rasa_url):
        self.rasa_url = rasa_url

    def get_intent(self, user_input):
        # Invia il messaggio a Rasa
        response = requests.post(f"{self.rasa_url}/model/parse", json={"text": user_input})
        response_data = response.json()

        # Ottieni l'intento
        return response_data.get("intent", {}).get("name", "unknown")
