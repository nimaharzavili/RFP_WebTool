from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import os, pickle
from rfpApp.engine.document_parser import DocumentParser
dp_obj = DocumentParser('pdf')
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
node_parser = MarkdownElementNodeParser(num_workers=8)
from rfpApp.engine.query_summarizer import paraphraseQuery
import asyncio
reranker = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)

async def home(request, *args, **kwargs):
    if request.method == "POST":
        query_text = request.POST["textbox"]
        file = request.FILES["file"]

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return HttpResponse("Invalid file type")
        df_json = df.to_json()
        print('I am within calculate rank view!')
        ranked_data_user = dp_obj.recursive_indexing(query_text)
        context = {"message": "success", 'task_id': ranked_data_user.id}
        
        return render(request, "home.html", context=context)
    elif request.method == "GET":
        return render(request, "home.html")
    else:
        return HttpResponse("Invalid Method")
    
def cor1(raw_index, query_text):
    recursive_query_engine = raw_index.as_query_engine(
                similarity_top_k=2, node_postprocessors=[reranker], verbose=True, temperature=0.7
            )
    return recursive_query_engine.aquery(query_text)
    
def cor2(raw_index, query_text):
    recursive_query_engine = raw_index.as_query_engine(
                similarity_top_k=2, node_postprocessors=[reranker], verbose=True, temperature=0.8
            )
    return recursive_query_engine.aquery(query_text)
    
async def directSearch(request):
    response_text1 = ""
    response_text2 = ""
    if request.method == 'POST':
        query_text = request.POST["textbox"]
        index_file_path = os.path.join('static', 'index.pkl')
        
        if not os.path.exists(index_file_path):
            return HttpResponse('Index file not found. Please first do the indexing, then you can start searching HGS products and services.')
        
        try:
            with open(index_file_path, 'rb') as index_file:
                raw_index = pickle.load(index_file)

            # recursive_query_engine = raw_index.as_query_engine(
            #             similarity_top_k=5, node_postprocessors=[reranker], verbose=True, temperature=0.9
            #         )
            
            response_1, response_2 = await asyncio.gather(*[cor1(raw_index, query_text),cor2(raw_index, query_text)])

            print("I passed")
            if hasattr(response_1, 'text') or hasattr(response_2, 'text'): 
                response_text1 = response_1.text
                response_text2 = response_2.text
            else:
                response_text1 = str(response_1)
                response_text2 = str(response_2)
            
            # print(f"modified resp:{response_text2}")
        except Exception as e:
            return HttpResponse(str(e))
        
    return render(request, "directSearch.html", {'response_text1': response_text1, 'response_text2': response_text2})

async def customSearch(request):
    return render(request, "customSearch.html")

# async def directSearch(request):
#     response_text = ""
#     if request.method == 'POST':
#         query_text = request.POST["textbox"]
#         index_file_path = os.path.join('static', 'index.pkl')
        
#         if not os.path.exists(index_file_path):
#             return HttpResponse('Index file not found. Please first do the indexing, then you can start searching HGS products and services.')
        
#         try:
#             with open(index_file_path, 'rb') as index_file:
#                 raw_index = pickle.load(index_file)


#             recursive_query_engine = raw_index.as_query_engine(
#                 similarity_top_k=5, node_postprocessors=[reranker], verbose=True
#             )

#             response = recursive_query_engine.query(query_text)

#             if hasattr(response, 'text'): 
#                 response_text = response.text
#             else:
#                 response_text = str(response)

#         except Exception as e:
#             return HttpResponse(str(e))
        
#     return render(request, "search.html", {'response_text': response_text})
