import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


tokenized_sent = []

with open ('./data/train.txt') as td:
  doc = td.readlines()
  stop_words = set(stopwords.words("english"))
  lem = WordNetLemmatizer()

  # print(doc)
  for sent in doc:
    sent_tuple = sent.split(';')
    tokenized_words = [word.lower() for word in word_tokenize(sent_tuple[0])]
    filtered_words = list(filter(lambda x: (x not in stop_words), tokenized_words))
    lemmatized_words=[lem.lemmatize(word) for word in filtered_words]
    tokenized_sent.append((lemmatized_words, sent_tuple[1].replace('\n', '')))

td.close()

print(tokenized_sent[:50])
for tuple in tokenized_sent:
  print(tuple[1])
emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

