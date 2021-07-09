from django.conf.urls import url
from django.urls import path, include
from . import views, Mail_sender




urlpatterns = [
    path('', views.home, name='home'),
    path('result/', views.save_keyword, name='save_kwd'),
    path('mail/', Mail_sender.mail_sender, name='mail_sender'),

]