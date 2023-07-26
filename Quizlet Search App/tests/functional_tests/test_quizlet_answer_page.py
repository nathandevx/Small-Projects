from selenium import webdriver
from quizlet_study_tool.models import QuizletStudyToolModel
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.urls import reverse
import time


class TestQuizletAnswerPage(StaticLiveServerTestCase):

	def setUp(self):
		self.browser = webdriver.Chrome(executable_path=r'main_project\tests\functional_tests\chromedriver.exe')

	def tearDown(self):
		self.browser.close()

	# Test if a form validation error occurred

