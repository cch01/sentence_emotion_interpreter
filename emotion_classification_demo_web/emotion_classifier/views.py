from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
import os
from django.views.decorators.csrf import csrf_exempt

#method for load pickled files from path
def load_variable_from_file(path):
  load_target = open(path, 'rb')
  variable = pickle.load(load_target)
  load_target.close() 
  return variable


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
classifier_path = os.path.join(THIS_FOLDER, 'emotion_nb_classifier.pickle')
tf_path = os.path.join(THIS_FOLDER, 'fitted_tf.pickle')

# initialize stop words and lemmatizer from nltk
stop_words = set(stopwords.words("english"))
lem = WordNetLemmatizer()

#load classifier and fitted tfidf vectorizor from file
nb_classifier = load_variable_from_file(classifier_path)
tf = load_variable_from_file(tf_path)


# web endpoint
def home(req):
  return render(req, 'emotion_classifier/home.html')


# POST api for accepting sentence input and return predicted emotion
@csrf_exempt
def predict_emotion(req):
  # method for preprocess text input sentence
  def input_preprocess(input):
    tokenized_words = [word.lower() for word in word_tokenize(input)]
    filtered_words = list(filter(lambda x: (x not in stop_words), tokenized_words))
    lemmatized_words=[lem.lemmatize(word) for word in filtered_words]
    return lemmatized_words

  # only accept POST request
  if req.method != 'POST':
    return HttpResponse(status=500)

  # return error if no input
  input = req.POST.get('input', '')
  if not input:
    return HttpResponse(status=400)

  tokenized_words = input_preprocess(input)
  # fit input into loaded vectorizer
  input_tf = tf.transform([' '.join(tokenized_words)])
  predicted_input_result = nb_classifier.predict(input_tf)
  return JsonResponse({'emotion': predicted_input_result[0]})