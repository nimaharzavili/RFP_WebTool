from django.urls import path
from . import views
from django.urls import path

app_name = "rfpApp"

urlpatterns = [
    path('', views.home, name='home'),
    path('directSearch/', views.directSearch, name='directSearch'),
    path('customSearch/', views.customSearch, name='customSearch'),
    path('download/<int:id>/', views.download, name='download')
    # path('download_results/<int:id>/', views.download_ranked_data, name='download_ranked_data')
]