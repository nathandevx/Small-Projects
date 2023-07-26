from django.test import SimpleTestCase
from quizlet_study_tool.forms import QuizletStudyToolModelForm


class TestForms(SimpleTestCase):

	def test_valid_data(self):
		model_form = QuizletStudyToolModelForm(data={
			'question': "What is a ribosome",
			'stop_words': False,
			'num_links': 3,
			'num_results': 3,
		})

		self.assertTrue(model_form.is_valid())

	def test_invalid_data(self):
		model_form = QuizletStudyToolModelForm(data={})

		self.assertFalse(model_form.is_valid())
