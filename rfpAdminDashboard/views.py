from django.shortcuts import render
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import os
from django.http import JsonResponse
from django.contrib import messages
from rfpAdminDashboard.embedding.embedding_engine import DocumentAnalysis

def dashboard(request, *args, **kwargs):
    if request.method == "POST":
        query_text = request.POST["textbox"]
        file = request.FILES["file"]

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return HttpResponse("Invalid file type")
        
        return render(request, "dashboard.html")
    elif request.method == "GET":
        return render(request, "dashboard.html")
    else:
        return HttpResponse("Invalid Method")

def fileManager(request, *args, **kwargs):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')

        if uploaded_files:
            try:
                for file in uploaded_files:
                    file_path = os.path.join('rfpAdminDashboard', 'data', 'RFP_files', file.name)
                    with open(file_path, 'wb') as f:
                        for chunk in file.chunks():
                            f.write(chunk)
                messages.success(request, 'Files uploaded successfully.')

            except Exception as e:
                messages.error(request, f'Error uploading files: {str(e)}')
        else:
            messages.warning(request, 'No files were selected for upload.')
    return render(request, 'fileManager.html')


def embedding(request, *args, **kwargs):
    if request.method == "POST":
        indexing_granularity = request.POST.get('bool_value', None)
        docTypesList = request.POST.getlist('items')
        dp_obj = DocumentAnalysis(useNode=bool(indexing_granularity), specificTypesList=docTypesList)
        dp_obj.indexing()
        return HttpResponse("Indexing is finished.")
    elif request.method == "GET":
        return render(request, "embedding.html")
    else:
        return HttpResponse("Invalid Method")