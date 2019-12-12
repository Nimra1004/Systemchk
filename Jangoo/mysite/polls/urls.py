from django.urls import path
from django.conf import settings

from django.conf.urls.static import static

from . import views
from polls import views

urlpatterns = [
    path('', views.home1, name='home1'),
    path('get-response/', views.get_response),
    #path('', views.Home.as_view(), name='Home'),
    #path('get_name', views.get_name),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

