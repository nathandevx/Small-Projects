"""
Gets the Google data
- Reads a text file for search phrases like 'best cheap gaming laptop'
- Uses the 'googlesearch' module to get the first 30 links of a given search phrase.
- Then I scrape those links for the H1 tag, title tag, domain name, and more.
- I then export the data as a CSV using pandas.

"""
from googlesearch import search
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from time import sleep
import requests
import pandas as pd


# This class gets the data.
class GetData:

    # Initializes everything.
    def __init__(self, query: str, num_of_results: int, pause_search: int = 10):
        """
        Initializes all the lists and variables.
        :param query: The search query. Ex: "best cheap gaming laptop".
        :param num_of_results: How many results you want. Ex: '30' for 30 links.
        :param pause_search: How many lapse to wait between HTTP requests. Only change it if you know what
        you're doing.
        """
        self.query = query
        self.num_of_results = num_of_results
        self.pause_search = pause_search

        # Domain name
        self.final_domain_names = []
        self.final_domain_names_length = []
        self.final_keywords_domain_names = []
        # Permalinks
        self.final_permalinks = []
        self.final_permalinks_length = []
        self.final_keywords_in_permalinks = []
        # H1 tag
        self.final_h1_tags = []
        self.final_h1_tag_length = []
        self.final_keywords_in_h1_tag = []
        # The page title
        self.final_titles = []
        self.final_keywords_in_title = []
        self.final_title_length = []
        # Keywords on page
        self.final_keywords_on_page = []
        # Characters on page
        self.final_character_count = []
        # The rank position
        self.final_rank_position = []

        # If 'only_standard' is False, the search will return every possible link it finds while searching
        self.results = []

    def get_links(self):
        results = [result for result in search(query=self.query,
                                               stop=self.num_of_results,
                                               pause=self.pause_search,
                                               only_standard=True)]
        self.results = results

    # Gets the domain names and permalinks
    def get_domain_names_and_perma_links(self):
        """
        Gets the domain name and the permalink. Finds the length of both the domain name and permalink.
        :return: Finalizes 'final_domain_names', 'final_domain_names_length',
        'final_permalinks', final_permalinks_length'
        """
        # Getting the domain names and permalinks
        for link in self.results:
            if 'http://www.' in link:
                # Gets the link after the 'http://www.'
                link = link[11:]
                # Where the dot in '.com' will be
                dot = link.index('.')
                # Adding what comes before the '.com' to the 'domain_names' list
                self.final_domain_names.append(link[:dot])
                self.final_domain_names_length.append(len(link[:dot]))
                # Adding what comes after the '.com' to the 'perma_links' list
                self.final_permalinks.append(link[dot:])
                self.final_permalinks_length.append(len(link[dot:]))
            # Same logic as above, except checking for 'https://www.'
            elif 'https://www.' in link:
                link = link[12:]
                dot = link.index('.')
                self.final_domain_names.append(link[:dot])
                self.final_domain_names_length.append(len(link[:dot]))

                self.final_permalinks.append(link[dot:])
                self.final_permalinks_length.append(len(link[dot:]))
            # If there is no 'www' in the link
            else:
                # Where the '://' is
                slash_slash = link.index('://') + 3
                # Where the dot in '.com' is
                dot_com = link.index('.')
                # Appending what comes after the '://' but before the dot in '.com' to the domain names
                self.final_domain_names.append(link[slash_slash:dot_com])
                self.final_domain_names_length.append(len(link[slash_slash:dot_com]))
                # Appending what comes after the dot in '.com' to the permalinks
                self.final_permalinks.append(link[dot_com:])
                self.final_permalinks_length.append(len(link[dot_com:]))

    # Finds the amount of keywords in the domain name and permalink.
    def keywords_in_domain_name_and_permalinks(self):
        """
        How many times do search query words appear in the domain name and permalink?
        :return: Finalizes 'final_keywords_domain_names' and 'final_keywords_in_permalinks'
        """
        def occurrences(link_list, search_query):
            """
            :param link_list: The list.
            :param search_query: The search query.
            :return: A list of the total occurrences of each word in the search query set in the provided list.
            """
            # List containing the occurrences
            final_list = []
            # How many times a search query word is in the domain name
            total = 0
            # Looping through the domain names
            for link in link_list:
                # Looping through each word in the search query
                for word in search_query.split():
                    # If that query word is in the domain name
                    if word in link:
                        # Increment total
                        total += 1
                # Once it's done looping through the words in the query, it appends the total to the keywords in domain
                # names list
                final_list.append(total)
                # Reinitialize total to 0
                total = 0
            return final_list

        self.final_keywords_in_permalinks = occurrences(self.final_permalinks, self.query)
        self.final_keywords_domain_names = occurrences(self.final_domain_names, self.query)

    # Gets the h1 tags.
    def get_h1_tags(self):
        """
        Gets the h1 tags on the page.
        :return: Finalizes 'final_keywords_in_h1_tag', 'final_h1_tag_length', 'final_h1_tags'
        """
        # The search query
        query = self.query.split()

        # Initializing total
        # Used to keep track of the number of words in the 'search query' that are
        # also in the 'h1 tag'
        total = 0
        # For each link in results
        for result in self.results:
            # Request access to the link
            try:
                request = requests.get(result)

                # If the status code is not 200 - I can't scrape it
                if request.status_code != 200:
                    self.final_keywords_in_h1_tag.append('unscrapable')
                    self.final_h1_tag_length.append('unscrapable')
                    self.final_h1_tags.append('unscrapable')
                # Otherwise, if I can access the link
                else:
                    # Create a soup object that will allow me to get the 'h1 tags'
                    soup = BeautifulSoup(request.text, 'html.parser')
                    # Getting the first occurrence of the 'h1 tag'
                    h1_tag = soup.find('h1')
                    # If No 'h1 tag' was found
                    if h1_tag is None:
                        self.final_keywords_in_h1_tag.append('no h1 tag')
                        self.final_h1_tag_length.append('no h1 tag')
                        self.final_h1_tags.append('no h1 tag')
                    # Otherwise if a 'h1 tag' was found
                    else:
                        self.final_h1_tags.append(h1_tag.text.lower())
                        # Get the 'h1 tag' text, lowercase it, make it a list
                        h1_tag = h1_tag.text.lower().split()
                        # For each word in the search query string list
                        # Ex: ['best', 'cheap', 'gaming', 'laptop']
                        for word in query:
                            # If the lowercase word is in the 'h1_tag' list
                            if word.lower() in h1_tag:
                                # Increment total by 1
                                total += 1
                        self.final_keywords_in_h1_tag.append(total)
                        self.final_h1_tag_length.append(len(' '.join(h1_tag)))

                    # Reinitializing 'total'
                    total = 0

            except:
                self.final_keywords_in_h1_tag.append('exception')
                self.final_h1_tag_length.append('exception')
                self.final_h1_tags.append('exception')

    # Gets the title tag.
    def get_titles(self):
        """
        Gets the title tag of a link
        :return: Finalizes 'final_keywords_in_title', 'final_title_length', 'final_titles'
        """
        # Initializes 'total' (the total number of times a search query word appeared in the 'title tag')
        total = 0
        # Iterating through the links
        for link in self.results:
            try:
                # Requesting access to the link
                request = requests.get(link)
                # If the status code is not 200 - I can't access the site
                if request.status_code != 200:
                    self.final_keywords_in_title.append('not 200')
                    self.final_title_length.append('not 200')
                    self.final_titles.append('ntnsoin 200')
                # Otherwise if I can access the link
                else:
                    # Create a soup object
                    soup = BeautifulSoup(request.text, 'html.parser')
                    # Get raw HTML title tag with text
                    title = soup.title
                    # If there is no title tag on the page
                    if title is None:
                        self.final_keywords_in_title.append('none')
                        self.final_title_length.append('none')
                        self.final_titles.append('none')
                    # Otherwise if there is a title tag
                    else:
                        # Make a list of the search query words
                        query_list = self.query.split()
                        # Get the text of the title tag and lowercase it
                        title = soup.title.text.lower()
                        self.final_titles.append(title)
                        # Make a list of words in the title tag
                        title_list = title.split()
                        # For each word in the search query
                        for word in query_list:
                            # If that word (lowercased) is in the title
                            if word.lower() in title_list:
                                # Increment total by 1
                                total += 1
                        self.final_keywords_in_title.append(total)
                        self.final_title_length.append(len(title))

                total = 0

            except:
                self.final_keywords_in_title.append('exception')
                self.final_title_length.append('exception')
                self.final_titles.append('exception')

    # Gets the page text.
    def get_page_text(self):
        """
        Gets the character count of a page.
        Gets the total number of times each word that is not a stopword from the 'search query' appears in the text.
        :return:
        """
        # Initializes 'total' (the total number of times a search query word appeared in the 'title tag')
        total = 0
        # Iterating through the links
        for link in self.results:
            try:
                # Requesting access to the link
                request = requests.get(link)
                # If the status code is not 200 - I can't access the site
                if request.status_code != 200:
                    self.final_keywords_on_page.append('nt 200')
                    self.final_character_count.append('not 200')
                # Otherwise if I can access the link
                else:
                    # Create a soup object
                    soup = BeautifulSoup(request.text, 'html.parser')
                    # Total number of characters on the page
                    character_count = len(soup.text)
                    # If the character count is 'None' or if it's zero (that means the page is empty)
                    if character_count is None or character_count <= 0:
                        self.final_keywords_on_page.append('none')
                        self.final_character_count.append('none')
                    # Otherwise if the character count is > 0
                    else:
                        # Loading the stopwords
                        stop_words = set(stopwords.words('english'))
                        # Make a list of the search query words and removing the 'stop words' to prevent words
                        # like 'the' 'a' 'of' from being counted
                        query_list = [word.lower() for word in self.query.split() if word.lower() not in stop_words]
                        # Gets the page text and lowercases it
                        page_text = soup.text.lower()
                        # For each word in the search query
                        for word in query_list:
                            total += page_text.count(' ' + word.lower() + ' ')
                        self.final_keywords_on_page.append(total)
                        self.final_character_count.append(character_count)

                total = 0

            except:
                self.final_keywords_on_page.append('exception')
                self.final_character_count.append('exception')

    # Where the site ranks in SERPS.
    def get_rank_position(self):
        """
        Finds out where the site ranks in SERPS.
        :return: An integer 1-30.
        """
        for index, link in enumerate(self.results):
            self.final_rank_position.append(index + 1)


# Loads the search queries file.
def load_file(file_name: str):
    """
    Loads the file containing the search queries.
    :param file_name: The file name that contains the search queries.
    :return: A list of search queries.
    """
    with open(file_name, 'r') as file:
        search_phrases = [line.rstrip('\n') for line in file]
    return search_phrases


# Makes the dataset.
def make_dataframe(load_file_name: str, make_file_name: str):
    """
    Loads the search queries, then makes a CSV file.
    :param load_file_name: The search queries file.
    :param make_file_name: The name to give the exported CSV file.
    :return: A pandas DataFrame.
    """
    # Loading the search queries
    search_queries = load_file(file_name=load_file_name)

    # Initializing the lists
    all_domain_names = []
    all_domain_name_lengths = []
    all_keywords_in_domain = []

    all_permalinks = []
    all_permalink_lengths = []
    all_keywords_in_permalink = []

    all_h1_tags = []
    all_h1_tag_lengths = []
    all_keywords_in_h1_tags = []

    all_title_tags = []
    all_title_tag_lengths = []
    all_keywords_in_title = []

    all_keywords_on_page = []
    all_character_counts = []

    all_rank_positions = []

    # Loops through the search queries
    for search_query in search_queries:
        sleep(1)
        data = GetData(query=search_query, num_of_results=30, pause_search=10)
        sleep(2)

        # Gets the links
        data.get_links()
        # The domain name and permalink
        data.get_domain_names_and_perma_links()
        all_domain_names += data.final_domain_names
        all_domain_name_lengths += data.final_domain_names_length
        all_permalinks += data.final_permalinks
        all_permalink_lengths += data.final_permalinks_length

        # Gets the total amount of keywords in the domain name and permalink
        data.keywords_in_domain_name_and_permalinks()
        all_keywords_in_permalink += data.final_keywords_in_permalinks
        all_keywords_in_domain += data.final_keywords_domain_names

        # H1 tags
        data.get_h1_tags()
        all_h1_tags += data.final_h1_tags
        all_h1_tag_lengths += data.final_h1_tag_length
        all_keywords_in_h1_tags += data.final_keywords_in_h1_tag

        # Title tags
        data.get_titles()
        all_title_tags += data.final_titles
        all_title_tag_lengths += data.final_title_length
        all_keywords_in_title += data.final_keywords_in_title

        # Total amount of keywords on the page and how many characters are on the page
        data.get_page_text()
        all_keywords_on_page += data.final_keywords_on_page
        all_character_counts += data.final_character_count

        # Where the page ranks
        data.get_rank_position()
        all_rank_positions += data.final_rank_position

    # Making the dataframe
    df = pd.DataFrame(data=zip(all_domain_names, all_domain_name_lengths, all_keywords_in_domain,
                               all_permalinks, all_permalink_lengths, all_keywords_in_permalink,
                               all_h1_tags, all_h1_tag_lengths, all_keywords_in_h1_tags,
                               all_title_tags, all_title_tag_lengths, all_keywords_in_title,
                               all_keywords_on_page, all_character_counts,
                               all_rank_positions),
                      columns=['Domain', 'Domain length', 'Keywords in domain',
                               'Permalink', 'Permalink length', 'Keywords in permalink',
                               'H1 tag', 'H1 tag length', 'Keywords in H1',
                               'Title tag', 'Title tag length', 'Keywords in title',
                               'Keywords on page', 'Character count',
                               'Rank position'])

    # Exporting the dataframe as a CSV
    df.to_csv(make_file_name)


# Makes the CSV file
make_dataframe('search_phrases.txt', 'serps_data.csv')
