import os
from langchain_mistralai import ChatMistralAI

class RFPModels:
    def __init__(self):
        os.environ["MISTRAL_API_KEY"] = "YuEIIt3ic04myFbvOMiMF3xIQaarajcy"
        
    def _mistral(self):
        model = ChatMistralAI(model="mistral-large-latest")
        return model
    





