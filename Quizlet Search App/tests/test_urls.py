from django.test import SimpleTestCase
from django.urls import reverse, resolve
from quizlet_study_tool.views import (
	quizlet_answer, remove_stopwords, get_links,
	get_questions_answers, get_results, run)


# Use 'SimpleTestCase' when you don't need to interact with the database
class TestUrls(SimpleTestCase):

	def test_quizlet_answer(self):
		# The url '/quizlet-answer/'
		url = reverse('quizlet_study_tool:quizlet-answer')
		# I think this checks if the URL is a 404
		resolve_url = resolve(url)
		print(resolve_url.func)
		# '.func' gets the view name - so it checks if the view names are the same?
		# do '.func.view_class' if the view is CBV
		self.assertEquals(resolve(url).func, quizlet_answer)

