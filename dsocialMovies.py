import nltk
import requests
import json
import collections
from nltk.metrics import precision, recall, f_measure
import random
from nltk.corpus import twitter_samples
from tabulate import tabulate

print(twitter_samples.fileids())

strings = twitter_samples.strings('tweets.20150430-223406.json')
for string in strings[:15]:
    print(string)

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

'''
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

data = []
response = requests.get(baseUrl + '/messages')
if(response.ok):
    data = json.loads(response.content.decode('utf-8'))
else:
    response.raise_for_status()

#start extract_features
def extract_features(msg):
    words = set(msg)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in words)
    return features

featuresets = [(extract_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

#extract feature vector for all messages
#n = int(len(pairs)*0.8)
#train_set = nltk.classify.util.apply_features(extract_features, pairs[:n])
#test_set = nltk.classify.util.apply_features(extract_features, pairs[n:])
#print('train on %d instances, test on %d instances' % (n, len(pairs)-n))

# Train the classifier
#table = []
#table.append(assess_classifier(nltk.NaiveBayesClassifier.train(train_set), test_set, "Naive Bayes"))
#table.append(assess_classifier(nltk.DecisionTreeClassifier.train(train_set), test_set, "Decision Tree"))
#table.append(assess_classifier(nltk.MaxentClassifier.train(train_set, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 4), test_set, "Maximum Entropy"))
#table.append(assess_classifier(nltk.NaiveBayesClassifier.train(train_set), test_set, "Support Vector Machines"))
#print(tabulate(table, headers=["Classifier", "Accuracy", "Precision(pos)", "Recall(pos)", "F-measure(pos)",
#                               "Precision(neg)", "Recall(neg)", "F-measure(neg)",
#                               "Precision(neutral)", "Recall(neutral)", "F-measure(neutral)"]))



#test the classifier
for message in data['messages']:
    print(message['nlp']['clean'] + " (" + classifier.classify(extract_features(message['nlp']['unigrams'])) + ")")

'''