''' Load libraries '''
import pandas as pd                                             # For Data Exploration, Manipulation                                              # For punctuation 
import numpy as np                                              # To create arrays
import nltk                                                     # For Text Pre-processing                   
from nltk.tokenize import word_tokenize                         # Tokenize text into words
from nltk.stem import PorterStemmer                             # Reducing word to it's root
from sklearn.feature_extraction.text import CountVectorizer     # Create Bag of Words
from sklearn.model_selection import train_test_split            # Split data into groups (Testing and Training)
from sklearn.naive_bayes import MultinomialNB                   # Selecting the Multinomial Algorithm 
from sklearn.metrics import accuracy_score                      # Display Accuracy 

from nltk.corpus import stopwords
from string import punctuation

# Use in case you get an error trying to import stopwords
# nltk.download('stopwords') 

'''
Import your data into the program and display it

Task: Load dataset and display dataset
Hint: Using pandas will make your life a lot easier
'''

'''
Check for any Null Values (empty rows) and drop duplicate rows

Task: Eliminate empty and duplicate rows
Hint: Use pandas!
'''

'''
Now it's time to start cleaning. Let's remove any unnecessary pieces of text.

Hint: Display one piece of text to see what we should remove
Task: Iterate over rows and perform cleaning, then display your dataset again
'''

'''
Create your final corpus of sentences. The corpus must be a list of all sentences
in its stemmed form and should not include punctuation characters or stopwords.

Task: Create a list of strings containing each stemmed and processed sentence.
Hint: Tokenize each sentence to handle words separately. Use word_tokenize to
tokenize and PortStemmer() to stem.
'''

'''
Create a Bag of Words representation of your corpus (x) and a list of the
labels (y). Both must have the same length!

Task: Create a Bag of Words model and its respective list of labels
Hint: Use scikit's CountVectorizer()
'''

'''
Split your data into a training set and a testing set. We chose 20% for the
test size, but you can tweak this value and see how it affects the final result.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

'''
Classify your data using Naive Bayes algorithm.

Task: Create a Naive Bayes classifier using only your training data (i.e. 
x_train and y_train).
Hint: Use scikit's MultinomialNB()
'''

'''
Measure the accuracy of your model with the testing data.

Task: Use your classifier to make predictions for x_test and then determine 
its accuracy in respect to y_test.
Hint: Use scikit's accuracy_score()
'''

'''
OPTIONAL
Make it accept user's input and determine whether or not the text entered is
spam.
'''

