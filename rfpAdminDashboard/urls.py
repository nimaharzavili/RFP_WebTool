from django.urls import path
from . import views
from django.urls import path

app_name = "rfpAdminDashboard"

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('embedding/', views.embedding, name='embedding'),
    path('fileManager/', views.fileManager, name='fileManager'),
    # path('download_results/<int:id>/', views.download_ranked_data, name='download_ranked_data')
]