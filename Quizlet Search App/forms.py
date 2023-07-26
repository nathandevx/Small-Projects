from django import forms
from .models import QuizletStudyToolModel


class QuizletStudyToolModelForm(forms.ModelForm):
	class Meta:
		model = QuizletStudyToolModel
		fields = ['question', 'stop_words', 'num_links', 'num_results']
		widgets = {
			'question': forms.TextInput(
				attrs={
					'placeholder': 'What is a mitochondria?',
					'class': 'form-control',
				}),
			'stop_words': forms.CheckboxInput(
				attrs={
					'class': 'form-check-label'
				}),
			'num_links': forms.NumberInput(
				attrs={
					'class': 'form-control',
				}),
			'num_results': forms.NumberInput(
				attrs={
					'class': 'form-control',
				}),
		}
