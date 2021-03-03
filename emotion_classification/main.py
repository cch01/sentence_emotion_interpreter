from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import random
import pickle

stop_words = set(stopwords.words("english"))
lem = WordNetLemmatizer()

# function for dumping variables to binary file for demo web
def dump_variable_to_file(variable, path):
  save_target = open(path, 'wb')
  pickle.dump(variable, save_target)
  save_target.close() 


# function for preprocessing train and validation (i.e. labelled) data
# return a list of tuples consist of a tokenized sentence and its label
def tokenize_labelled_data_from_txt(txt_path):
  tokenized_sentences = []
  with open (txt_path) as text_data:
    doc = text_data.readlines()
    for sentence in doc:
      sentence_tuple = sentence.split(';')
      tokenized_words = [word.lower() for word in word_tokenize(sentence_tuple[0])]
      filtered_words = list(filter(lambda x: (x not in stop_words), tokenized_words))
      lemmatized_words=[lem.lemmatize(word) for word in filtered_words]
      tokenized_sentences.append((lemmatized_words, sentence_tuple[1].replace('\n', '')))
  text_data.close()
  return tokenized_sentences


# function for preprocessing test data
# return a list of tokenized sentences
def tokenize_test_data_from_txt(txt_path):
  tokenized_test_sentences = []
  with open ('./data/test_data.txt') as test_data:
    doc = test_data.readlines()
    for sentence in doc:
      tokenized_words = [word.lower() for word in word_tokenize(sentence)]
      filtered_words = list(filter(lambda x: (x not in stop_words), tokenized_words))
      lemmatized_words=[lem.lemmatize(word) for word in filtered_words]
      tokenized_test_sentences.append(lemmatized_words)
    test_data.close()
  return tokenized_test_sentences


tokenized_train_sentences = tokenize_labelled_data_from_txt('./data/train.txt')

tokenized_validation_sentences = tokenize_labelled_data_from_txt('./data/val.txt')

emotions = ['fear', 'anger', 'joy', 'love', 'sadness', 'surprise']

tf = TfidfVectorizer()

# fit and transform train data
train_text_tf = tf.fit_transform([' '.join(tuple[0]) for tuple in tokenized_train_sentences])
train_text_label = [tuple[1] for tuple in tokenized_train_sentences]

# Generate model using Multinomial Naive Bayes
nb_classifier = MultinomialNB().fit(train_text_tf, train_text_label)

# dump the trained classifier and tfidf vectorizor to binary for web use
dump_variable_to_file(nb_classifier, 'emotion_nb_classifier.pickle')
dump_variable_to_file(tf, 'fitted_tf.pickle')

# transform validation data to be predicted
validation_text_tf = tf.transform([' '.join(tuple[0]) for tuple in tokenized_validation_sentences])
validation_text_label = [tuple[1] for tuple in tokenized_validation_sentences]

predicted_validation_result= nb_classifier.predict(validation_text_tf)

# compare the predicted result from validation data to the validation label
print("MultinomialNB Accuracy:", accuracy_score(validation_text_label, predicted_validation_result))
print(classification_report(validation_text_label, predicted_validation_result))

#import and preprocess the test data
tokenized_test_sentences = tokenize_test_data_from_txt('./data/test_data.txt')

# transform the test data to be predicted
test_text_tf = tf.transform([' '.join(tokenized_sentence) for tokenized_sentence in tokenized_test_sentences])

predicted_test_result = nb_classifier.predict(test_text_tf)

# write the predicted result into txt
writer = open('text_prediction.txt', 'w+')
writer.writelines('\n'.join(predicted_test_result))
writer.close()

