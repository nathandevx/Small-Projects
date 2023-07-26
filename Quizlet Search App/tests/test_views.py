from django.test import TestCase, Client
from django.urls import reverse
from quizlet_study_tool.models import QuizletStudyToolModel


class TestViews(TestCase):

	def setUp(self):
		self.client = Client()
		self.quizlet_answer_url = reverse('quizlet_study_tool:quizlet-answer')
		self.template_name = 'quizlet_study_tool/quizlet_answer.html'
		self.search1 = QuizletStudyToolModel.objects.create(
			question="What is a mitochondria?",
			stop_words=False,
			num_links=3,
			num_results=3,
		)

	def test_quizlet_answer_GET(self):
		response = self.client.get(self.quizlet_answer_url)

		self.assertEquals(response.status_code, 200)
		self.assertTemplateUsed(response, self.template_name)

	def test_quizlet_answer_POST(self):
		# Sends data
		response = self.client.post(self.quizlet_answer_url, {
			'question': 'Who was the president during the Civil War?',
			'stop_words': True,
			'num_links': 5,
			'num_results': 10,
		})

		self.assertEquals(response.status_code, 200)

	def test_quizlet_answer_POST_no_data(self):
		# Does not send data
		response = self.client.post(self.quizlet_answer_url)

		self.assertEquals(response.status_code, 200)























