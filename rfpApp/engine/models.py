import os
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

class RFPModels:
    def __init__(self):
        os.environ["MISTRAL_API_KEY"] = os.getenv('MISTRAL_API_KEY')
        
    def _mistral(self):
        model = ChatMistralAI(model="mistral-large-latest")
        return model
    





