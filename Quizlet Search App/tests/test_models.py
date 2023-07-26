from django.test import TestCase
from quizlet_study_tool.models import QuizletStudyToolModel


class TestModels(TestCase):
	def setUp(self):
		self.search1 = QuizletStudyToolModel.objects.create(
			question="What is a mitochondria?",
			stop_words=False,
			num_links=3,
			num_results=3,
		)



	# 	You could test that a user's name is automatically capitalized after creation
	def test_something(self):
		# self.assertEquals(self.search1.name, 'First Last')
		pass

