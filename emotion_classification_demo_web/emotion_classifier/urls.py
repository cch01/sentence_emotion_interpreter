from django.urls import path, include
from . import views
urlpatterns = [
    path('', views.home),
    path('predict_emotion', views.predict_emotion)
]
