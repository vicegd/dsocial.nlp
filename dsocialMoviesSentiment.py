import nltk
import requests
import json
import collections
from nltk.metrics import precision, recall, f_measure
import random
from nltk.corpus import movie_reviews
from tabulate import tabulate

baseUrl = "http://localhost:4001/api"

def assess_classifier(classifier, test_set, text):
    accuracy = nltk.classify.accuracy(classifier, test_set)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    pos_pre = precision(refsets['positive'], testsets['positive'])
    pos_rec = recall(refsets['positive'], testsets['positive'])
    pos_fme = f_measure(refsets['positive'], testsets['positive'])
    neg_pre = precision(refsets['negative'], testsets['negative'])
    neg_rec = recall(refsets['negative'], testsets['negative'])
    neg_fme = f_measure(refsets['negative'], testsets['negative'])
    neu_pre = precision(refsets['neutral'], testsets['neutral'])
    neu_rec = recall(refsets['neutral'], testsets['neutral'])
    neu_fme = f_measure(refsets['negative'], testsets['neutral'])

    return [text, accuracy, pos_pre, pos_rec, pos_fme, neg_pre, neg_rec, neg_fme,
                            neu_pre, neu_rec, neu_fme]

data = []
pairs = []
response = requests.get(baseUrl + '/messages')
if(response.ok):
    data = json.loads(response.content.decode('utf-8'))
    for message in data['messages']:
        if(message['nlp']['sentiment'] != 'unknown'):
            pairs.append((message['nlp']['unigrams'], message['nlp']['sentiment']))
else:
    response.raise_for_status()

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

tweet = 'Larry is my friend'
print(classifier.classify(extract_features(tweet.split())))
