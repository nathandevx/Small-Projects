# Django imports
from django.shortcuts import render
from django import template
from .forms import QuizletStudyToolModelForm

# Other package imports
from googlesearch import search
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import requests

MY_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"

def quizlet_answer(request):
	form = QuizletStudyToolModelForm()
	question = ""
	access = ""
	no_answers = ''
	stop_words = ''
	results = {}
	if request.method == 'POST':
		form = QuizletStudyToolModelForm(request.POST or None)
		if form.is_valid():
			question = form.cleaned_data['question']
			stop_words = form.cleaned_data['stop_words']
			num_links = form.cleaned_data['num_links']
			num_results = form.cleaned_data['num_results']
			results = run(question, num_links, num_results, stop_words)
			# form.save()
			no_answers = "No answers found"
			form = QuizletStudyToolModelForm(request.POST or None)
		else:
			access = "Something went wrong"
	else:
		access = "Welcome to the quizlet study tool."
		form = QuizletStudyToolModelForm(request.POST or None)

	template_name = 'quizlet_study_tool/quizlet_answer.html'
	context = {
		'access': access,
		'form': form,
		'question': question,
		'question_stopwords': remove_stopwords(question),
		'results': results,
		'no_answers': no_answers,
		'stop_words': stop_words,
	}
	return render(request, template_name, context)


# Removes stop words
def remove_stopwords(string: str):
	"""
	Removes all the stop words in a string
	:param string: the string
	:return: a new containing no stop words

	"""

	stop_words = set(stopwords.words('english'))
	return ' '.join([word for word in string.split() if word not in stop_words])


# Gets quizlet links
def get_links(search_phrase: str, num_of_links: int = 1):
	"""
	Gets the quizlet links from search.
	:param search_phrase: The search query.
	:param num_of_links: Number of links to get.
	:return: A list of links. OR 'None' if no 'quizlet' results are found.
	"""
	user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
	search_phrase = search_phrase + " quizlet"
	# 	domains=['quizlet.com'] searches for the first occurrence of 'quizlet' - not the first however many
	links = [result for result in search(query=search_phrase, stop=num_of_links, pause=5, user_agent=user_agent) if 'quizlet.com' in result]
	if links:
		return links[:num_of_links]
	else:
		return None


# Returns a list of questions and answers
def get_questions_answers(links: list):
	"""
	Gets all the questions and answers from the quizlet links
	:param links: The quizlet links.
	:return: One tuple containing 2 lists ([1, 2, 3], [4, 5, 6])

	"""
	if not links:
		return None

	questions = []
	answers = []
	quizlet_links = []
	headers = {
		'User-Agent': MY_USER_AGENT,
	}
	# Loops through the quizlet links
	for link in links:
		access = requests.get(link, headers=headers)
		# If the status code is not a 200, it means the crawler cannot access the site
		if access.status_code != 200:
			# I return None so the program doesn't crash
			return None
		soup = BeautifulSoup(access.text, 'html.parser')
		# questions_answers = [question, answer, question, answer, question] etc
		questions_answers = soup.find_all('div', attrs={'class', 'SetPageTerm-sideContent'})
		# I changed it
		# questions_answers = soup.find_all('span', attrs={'class', 'TermText notranslate lang-en'})
		# Appending 'question_answer' to either the 'questions' list or the 'answers' list
		for index, question_answer in enumerate(questions_answers):
			# The even indexes are the questions
			if index % 2 == 0:
				questions.append(question_answer.text)
				quizlet_links.append(link)
			else:
				answers.append(question_answer.text)
	return questions, answers, quizlet_links


# Gets the final results
def get_results(user_question: str, remove_stop_words: bool, num_of_links: int, num_of_results: int = 1):
	"""
	Finds the questions that related most closely to the user's question.
	:param user_question: The user's question.
	:param remove_stop_words: True = removes stopwords, False = do not remove stopwords
	:param num_of_links: The number of links the user wants to look through.
	:param num_of_results: The number of results the user wants to get printed out
	:return: a dictionary containing the top 5 (or whatever num_of_results is) results

	"""
	# Removes the stopwords from the user's question if 'remove_stop_words' is True
	if remove_stop_words:
		user_question = remove_stopwords(string=user_question)
	else:
		pass

	# The results dictionary, results = {similarity ratio: (question, answer)}
	results = {}

	# Checking if 'get_questions_answers()' returns None
	# Get questions answered and get links are stored into a variable so that they do not execute each time they are
	# called. Saving time.
	questions_answers = get_questions_answers(get_links(search_phrase=user_question, num_of_links=num_of_links))
	if questions_answers is None:
		return None
	else:
		pass

	# These are all parallel lists
	# Loads the questions list
	questions = questions_answers[0]
	# Loads the answers list
	answers = questions_answers[1]
	# Loads the links list
	links = questions_answers[2]

	# Looping through the 'questions' and 'answers' list
	for question, answer, link in zip(questions, answers, links):
		# If the user wants to remove stopwords, stopwords are removed from the question and answer
		if remove_stop_words:
			# Removing stopwords
			question_no_stopwords = remove_stopwords(string=question)
			answer_no_stopwords = remove_stopwords(string=answer)

			# ***Sometimes the answer is the question on quizlet.***
			# The similarity ratio is a float
			# Calculating the similarity between the user's question and the ANSWER on quizlet
			similarity_ratio_answer = SequenceMatcher(None, a=user_question, b=answer_no_stopwords).ratio()
			# Calculating the similarity between the user's question and the QUESTION on quizlet
			similarity_ratio_question = SequenceMatcher(None, a=user_question, b=question_no_stopwords).ratio()

			# If the quizlet question is more similar to the user's question, then that's the question
			if similarity_ratio_question > similarity_ratio_answer:
				# Adding to the results dictionary. results = {.553: 'question', 'answer'}
				results[similarity_ratio_question] = question, answer, link

			# If the quizlet answer is more similar to the user's question, then that's the question
			elif similarity_ratio_answer > similarity_ratio_question:
				results[similarity_ratio_answer] = answer, question, link
			# Otherwise (if both the question and answer's similarity ratios are the same)

			else:
				results[similarity_ratio_question] = question, answer, link

		# If the user does not want to remove stopwords
		# Same process as above. However the stopwords are not removed.
		else:
			similarity_ratio_answer = SequenceMatcher(None, a=user_question, b=answer).ratio()
			similarity_ratio_question = SequenceMatcher(None, a=user_question, b=question).ratio()

			if similarity_ratio_question > similarity_ratio_answer:
				results[similarity_ratio_question] = question, answer, link
			elif similarity_ratio_answer > similarity_ratio_question:
				results[similarity_ratio_answer] = answer, question, link
			else:
				results[similarity_ratio_question] = question, answer, link

	# The final results
	results = sorted(results.items(), reverse=True)[:num_of_results]

	# Formatting the keys (95.23% instead of .9523481668)
	new_results = {}
	for key, value in results:
		new_key = str(round(key * 100, 2)) + '%'
		new_results[new_key] = value
	return new_results


# Runs the program
def run(question: str, num_links: int, num_results: int, stop_words: bool = False):
	"""
	Asks the user for input and runs the program.
	:return: Returns a dictionary that contains the final output.

	"""
	final_results = get_results(
		user_question=question,
		remove_stop_words=stop_words,
		num_of_links=num_links,
		num_of_results=num_results
	)

	# Checking if 'final_results' is None
	if final_results is None:
		return None
	else:
		pass

	return final_results


