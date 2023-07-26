from django.urls import path
from .views import quizlet_answer

app_name = 'quizlet_study_tool'
urlpatterns = [
	path('quizlet-answer/', quizlet_answer, name='quizlet-answer'),
]
