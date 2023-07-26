from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator


class QuizletStudyToolModel(models.Model):
	question = models.CharField(max_length=1000)
	stop_words = models.BooleanField()
	num_links = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(5)])
	num_results = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(10)])

	def __str__(self):
		return self.question

	class Meta:
		verbose_name = 'Question'
		verbose_name_plural = "Questions"



