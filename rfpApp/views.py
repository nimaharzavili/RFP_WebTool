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
# from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5)
# reranker = FlagEmbeddingReranker(top_n=5,model="BAAI/bge-reranker-large",)
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
                similarity_top_k=5, node_postprocessors=[reranker], verbose=True, temperature=0.7
            )
    return recursive_query_engine.aquery(query_text)
    
def cor2(raw_index, query_text):
    recursive_query_engine = raw_index.as_query_engine(
                similarity_top_k=5, node_postprocessors=[reranker], verbose=True, temperature=0.8
            )
    return recursive_query_engine.aquery(query_text)

def download(request, *args, **kwargs):
    record_id_str = kwargs.get("id", None)
    if not record_id_str:
        return HttpResponse("No ID Provided.", status=400)
    else:
        record_id = int(record_id_str)        
        if os.path.exists(f"seoanalysisapp/static/ranked_data_{record_id}.csv"):
            with open(f"seoanalysisapp/static/ranked_data_{record_id}.csv", "rb") as f:
                response = HttpResponse(f.read(), content_type="text/csv")
                response["Content-Disposition"] = f'attachment; filename="ranked_data_{record_id}.csv"'
                os.remove(f"seoanalysisapp/static/ranked_data_{record_id}.csv")
            return response
            
        else:
            return HttpResponse("File not found.", status=404)
        
async def directSearch(request):
    response_text1 = ""
    response_text2 = ""
    meta1 = []
    meta2 = []
    if request.method == 'POST':
        query_text = request.POST["textbox"]
        target_indexing = request.POST.get('option')

        index_file_path = os.path.join("static",target_indexing, f"{target_indexing}_index.pkl")
        
        if not os.path.exists(index_file_path):
            return HttpResponse('Index file not found. Please first do the indexing, then you can start searching HGS products and services.')
        
        try:
            with open(index_file_path, 'rb') as index_file:
                raw_index = pickle.load(index_file)

            # recursive_query_engine = raw_index.as_query_engine(
            #             similarity_top_k=5, node_postprocessors=[reranker], verbose=True, temperature=0.9
            #         )
            
            response_1, response_2 = await asyncio.gather(*[cor1(raw_index, query_text), cor2(raw_index, query_text)])

            desired_keys = ['page_num', 'paper_path', 'section_id', 'sub_section_id']
            items = [v for k, v in response_1.metadata.items()]
            meta1 = [items[0][key] for key in desired_keys if key in items[0]]
            meta2 = [items[1][key] for key in desired_keys if key in items[1]]

            if hasattr(response_1, 'text') or hasattr(response_2, 'text'): 
                response_text1 = response_1.text
                response_text2 = response_2.text
            else:
                response_text1 = str(response_1)
                response_text2 = str(response_2)
            
        except Exception as e:
            return HttpResponse(str(e))
        
    return render(request, "directSearch.html", {'response_text1': response_text1, 'response_text2': response_text2, 'meta1':meta1, 'meta2':meta2})

async def customSearch(request):
    return render(request, "customSearch.html")


