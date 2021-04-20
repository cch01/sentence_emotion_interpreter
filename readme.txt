# Simple sentence sentiment detector

Training a naive bayes model with tf-dif.
The model trained is used on a django webpage to predict sentiment of sentences.

Prerequisites python packages: `nltk`, `sklearn`, `django`

1. Train, validate and generate test result
`
cd emotion_classification
python3 main.py
`

2. Startup web UI server
* Need internet access to download the js scripts / css for html
`
cd emotion_classification_demo_web
python3 manage.py runserver
`
Open the browser, navigate to http://localhost:8000 to access the demo webpage.
