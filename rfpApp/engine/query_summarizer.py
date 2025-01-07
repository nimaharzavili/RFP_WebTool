from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rfpApp.engine.models import RFPModels

model = RFPModels()
model_obj = model._mistral()

def paraphraseQuery(user_query):
    messages = [
        (
            "system",
            "You are an AI assistant capable of paraphrasing questions, sentences, or paragraphs. Your task is to paraphrase the following text exactly. Do not provide any extra comments, explanations, or responses. Just paraphrase the text, and output the paraphrased version only.",
        ),
        ("human", user_query),
    ]
    
    new_query = model_obj.invoke(messages)
    return new_query.content