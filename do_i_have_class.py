"""
***** Have this program and the .txt file be in the same directory (folder) *****

This program reads in a text file that contains:
line: math 12, 9:00am-9:45am, monday wednesday thursday friday

Then tells the user when they have class.

"""

import datetime
import os.path


# Gets the classes
def read_file(file_name: str):
    """
    Reads the text file containing the information.

    :param file_name: The file path.
    :return: A dictionary containing the class name as the key. As wells as a tuple as the value that contains the class
    time and days of the week that the person has that class.

    """

    # The file contents
    file_contents_dict = {}
    # Days of the week list
    day_of_week_list = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    # Reads the file
    with open(file=file_name, mode='r') as file:
        # Reads each line in the file
        for line in file:
            # Removing '\n' from the end of each line, then creating a list separated by commas
            line = line.rstrip('\n').split(',')
            # Making a list of all the class's days
            day_of_week = line[2].lower().split()
            # Converting the days of week (Monday, Thursday, Sunday, etc) to an integer of 0-6. (0 being Monday,
            # 6 being Sunday)
            day_of_week = [day_of_week_list.index(day) for day in day_of_week]
            # dictionary = {'class_name': ('9:00am-9:45am', [0, 2, 4]), 'History 17B': ('5:00pm-6:30pm', [0, 4])}
            file_contents_dict[line[0]] = line[1].replace(' ', ''), day_of_week
    # Returns the dictionary
    return file_contents_dict


# Is the current time between 9:00am-10:00am?
def between_times(current_time=None, begin_time=None, end_time=None, before_time: bool = None):
    """
    Determines if the current time is between 2 times.

    :param current_time: The current time.
    :param begin_time: Start time.
    :param end_time: End time.
    :param before_time: Optional parameter. In case the user wants to determine if the current time is less than or
    equal to the beginning time. (Used to find out which classes have already passed.)
    :return: Boolean. If "before_time" is 'True', the function returns 'True' if the current time is less than or equal
    to the "begin_time", else it returns 'False'. Otherwise if "before_time" is left as 'None', the function returns
    true if the "current_time" is between or equal to "begin_time" and "end_time".

    """

    # If "before_time" is set to 'True'
    if before_time:
        begin_time = to_time_object(string=begin_time)
        return current_time <= begin_time
    # Otherwise if "before_time" is 'None'
    else:
        begin_time = to_time_object(string=begin_time)
        end_time = to_time_object(string=end_time)
        # Let's say current_time = 9:20am, begin_time = 9:00am, end_time = 10:00am. The function would return true
        # because "current_time" is in between "begin_time" and "end_time".
        return current_time >= begin_time and current_time <= end_time


# Converts a string to a datetime.time object.
def to_time_object(string: str):
    """
    Converts a string to a datetime.time object.

    :param string: The time string. For example: "3:40pm"
    :return: The military time version of the string as a datetime.time object. For example: '15:40:00'

    """

    return datetime.datetime.strptime(string, '%I:%M%p').time()


# Finds out if there is a class right now. As well as the classes left today.
def todays_classes(class_dictionary: dict):
    """
    This function finds out if there is a class right now and finds all the classes that the user has left today.

    :param class_dictionary: A dictionary containing {'class name': ('class time', [days of the week the class
    is on])}.
    For example: {'history 17b': ('10:00am-10:45am', [0, 2, 4]), 'philosophy 103': ('10:30am-11:45am', [1, 3])}
    :return: A tuple containing: (a string containing the class right now, list of classes left today).

    """

    # The current time
    current_time = to_time_object(datetime.datetime.now().strftime('%I:%M%p'))
    # The current day of the week integer (0 = Monday and 6 = Sunday)
    current_day_of_week = datetime.datetime.now().date().weekday()

    # Will contain todays classes
    today_classes = []
    # Initializing the variable
    class_right_now = ""

    # Finding out which classes are still left and if there is any class right now
    for class_name, value in class_dictionary.items():
        # A list containing the class time ['9:40am', '11:20am']
        class_time_list = value[0].split('-')
        # The days that the class is on
        class_day_list = value[1]
        # If the current weekday is in the "class_day_list" that means the class occurs today
        if current_day_of_week in class_day_list:
            # What time the class starts, like '9:40am'
            begin_time = class_time_list[0]
            # What time the class ends, like '10:40am'
            end_time = class_time_list[1]
            # True if the class is going on right now
            class_now = between_times(current_time=current_time, begin_time=begin_time, end_time=end_time)
            # True if the class has not passed (True: class is at 3PM and it is 2PM right now)
            the_class_has_not_passed = between_times(current_time=current_time, begin_time=begin_time, before_time=True)
            # If the class is right now, set it equal to the class_right_now variable
            if class_now:
                class_right_now = class_name
            # Else if if the class is later on today, append it to the "today_classes" list
            elif the_class_has_not_passed:
                today_classes.append((class_name, class_time_list))
            # Otherwise pass
            else:
                pass
        # Otherwise pass
        else:
            pass
    # Returns a string containing the class right now and a list containing the classes still left for today
    return class_right_now, today_classes


# Finds the class right now and how much time before other classes.
def find_next_class(classes: tuple):
    """
    Finds the time until each of todays classes.

    :param classes: A tuple containing (class right now, todays classes)
    :return: A tuple containing: (class right now, {'todays_class': hours until class, minutes until class}

    """

    # Gets the time and formats it
    now = datetime.datetime.now().strftime('%I:%M%p')
    # Converts the "now" into military time
    current_time = datetime.datetime.strptime(now, '%I:%M%p')
    # The class right now, if there are none then it is an empty string
    class_right_now = classes[0]

    # Finds the time until each class in the "classes" tuple
    time_until_class = {}
    for class_ in classes[1]:
        # Converting the class's time into a 24 hour datetime object
        starts_at = datetime.datetime.strptime(class_[1][0], '%I:%M%p')
        # Subtracting the time the class starts from the current time and getting the total seconds
        difference = (starts_at - current_time).total_seconds()
        # How many minutes until that class
        minutes = int(abs(divmod(difference, 60)[0]))
        # Getting the hours until the class
        hours = minutes // 60
        # Getting the minutes until the class
        minutes = minutes % 60
        # dictionary = {'class_name': (hours until class, minutes until class)}
        time_until_class[class_[0]] = hours, minutes

    # Returns the class right now and the time until each class
    return class_right_now, time_until_class


# Runs the program
def run_program():
    """
    Asks the user for input and runs the program.
    :return: Nothing.

    """

    # Tells the user the rules of the program.
    print("Create a text file containing your classes. Format it as such exactly --->\n")
    print("math 12, 9:00am-9:45am, monday wednesday thursday friday\n")
    print("Class name, class time, class days.\n")
    print("Start on a new line every time you want to create a new class.")
    print("Once you are done entering in your classes in the text file, save it in the same directory as this program"
          " and name it 'classes.txt'.\n")

    # What the file name should be
    file_name = "classes.txt"
    # Verifying that this file exists.
    while file_name != 'classes.txt' or not os.path.isfile(file_name):
        # Telling the user why the file name was not accepted.
        print("***File does not exist. You may have entered the name wrong, named it wrong, or the file is not in"
              " the same directory as this program.***\n")
        # Asking the user for the file name again.
        file_name = input("Enter the file name: ")

    # The final result tuple containing the class right now (if any) and a dictionary of all the classes left for today.
    final_answer = find_next_class(classes=todays_classes(class_dictionary=read_file(file_name=file_name)))

    # For better output
    print("-------------------------------")

    # If there is no class right now.
    if not final_answer[0]:
        print("\nYou have no classes right now.\n")
    # Otherwise if there is class right now.
    else:
        print(f"You have \"{final_answer[0]}\" right now.\n")

    # If there are no more classes today.
    if not final_answer[1]:
        print("No more classes today.")
    # Otherwise pass
    else:
        pass

    # For each class left today, print the class name, hours and minutes left until the class.
    for class_name_, time_until_class_ in final_answer[1].items():
        print(f"You have \"{class_name_}\" in {time_until_class_[0]} hours and {time_until_class_[1]} minutes.")

    # For better output
    print("-------------------------------")


if __name__ == "__main__":
    run_program()
