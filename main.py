from nltk.tokenize import word_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

tokenized_sent = []

with open ('./data/train.txt') as td:
  doc = td.readlines()
  stop_words = set(stopwords.words("english"))
  lem = WordNetLemmatizer()

  for sent in doc:
    sent_tuple = sent.split(';')
    tokenized_words = [word.lower() for word in word_tokenize(sent_tuple[0])]
    filtered_words = list(filter(lambda x: (x not in stop_words), tokenized_words))
    lemmatized_words=[lem.lemmatize(word) for word in filtered_words]
    tokenized_sent.append((lemmatized_words, sent_tuple[1].replace('\n', '')))

td.close()

# print(tokenized_sent[:50])
emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

tf=TfidfVectorizer()
text_tf= tf.fit_transform([' '.join(tuple[0]) for tuple in tokenized_sent])

X_train, X_test, y_train, y_test = train_test_split(text_tf, [tuple[1] for tuple in tokenized_sent], test_size=0.3)


# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print('predicted', predicted[:200])
print('y_test', y_test[:200])
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
# print(metrics.)

