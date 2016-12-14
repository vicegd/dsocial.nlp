import collections
import json
import nltk
import requests
import random
import lib.dsocialLib as dsocial
from nltk.metrics import precision, recall, f_measure
from tabulate import tabulate

baseUrl = "http://localhost:4001/api"
percent = 0.9

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

    return [text, accuracy, pos_pre, pos_rec, pos_fme, neg_pre, neg_rec, neg_fme]

data = []

pairsBagOfWords = []
featureListBagOfWords = []

featureListBagOfWordsRep = []

pairsUnigrams = []
featureListUnigrams = []

featureListBigrams = []
pairsBigrams = []

featureListTrigrams = []
pairsTrigrams = []

response = requests.get(baseUrl + '/messages')
if(response.ok):
    data = json.loads(response.content.decode('utf-8'))
    for message in data['messages']:
        if(message['nlp']['sentiment'] != 'unknown'):
            processedMessage = dsocial.processMessage(message['nlp']['source'])

            featureVectorBagOfWords = dsocial.getFeatureVectorUnigrams(processedMessage)
            featureListBagOfWords.extend(featureVectorBagOfWords)
            pairsBagOfWords.append((featureVectorBagOfWords, message['nlp']['sentiment']))

            featureVectorUnigrams = dsocial.getFeatureVectorUnigrams(processedMessage)
            featureListUnigrams.extend(featureVectorUnigrams)
            pairsUnigrams.append((featureVectorUnigrams, message['nlp']['sentiment']))

            featureVectorBigrams = dsocial.getFeatureVectorBigrams(processedMessage)
            featureListBigrams.extend(featureVectorBigrams)
            pairsBigrams.append((featureVectorBigrams, message['nlp']['sentiment']))

            featureVectorTrigrams = dsocial.getFeatureVectorTrigrams(processedMessage)
            featureListTrigrams.extend(featureVectorTrigrams)
            pairsTrigrams.append((featureVectorTrigrams, message['nlp']['sentiment']))

else:
    response.raise_for_status()

freqBagOfWords = nltk.FreqDist(featureListBagOfWords)
featureListBagOfWords = []
for element in freqBagOfWords.most_common(100):
    featureListBagOfWords.append(element[0])

for element in freqBagOfWords.most_common():
    if (element[1] > 1):
        featureListBagOfWordsRep.append(element[0])

featureListUnigrams = list(set(featureListUnigrams))
featureListBigrams = list(set(featureListBigrams))
featureListTrigrams = list(set(featureListTrigrams))

def extract_featuresBagOfWords(message):
    message_words = set(message)
    features = {}
    for word in featureListBagOfWords:
        features['contains(%s)' % word] = (word in message_words)
    return features

def extract_featuresBagOfWordsRep(message):
    message_words = set(message)
    features = {}
    for word in featureListBagOfWordsRep:
        features['contains(%s)' % word] = (word in message_words)
    return features

def extract_featuresUnigrams(message):
    message_words = set(message)
    features = {}
    for word in featureListUnigrams:
        features['contains(%s)' % word] = (word in message_words)
    return features

def extract_featuresBigrams(message):
    message_words = set(message)
    features = {}
    for word1, word2 in featureListBigrams:
        features['contains(%s,%s)' % (word1, word2)] = ((word1, word2) in message_words)
    return features

def extract_featuresTrigrams(message):
    message_words = set(message)
    features = {}
    for word1, word2, word3 in featureListTrigrams:
        features['contains(%s,%s,%s)' % (word1, word2, word3)] = ((word1, word2, word3) in message_words)
    return features

def extract_featuresUniBigrams(message):
    message_words = set(message)
    features = {}
    for word1 in featureListUnigrams:
        features['contains(%s)' % (word1)] = ((word1) in message_words)
    for word1, word2 in featureListBigrams:
        features['contains(%s,%s)' % (word1, word2)] = ((word1, word2) in message_words)
    return features

def extract_featuresBiTrigrams(message):
    message_words = set(message)
    features = {}
    for word1, word2 in featureListBigrams:
        features['contains(%s,%s)' % (word1, word2)] = ((word1, word2) in message_words)
    for word1, word2, word3 in featureListTrigrams:
        features['contains(%s,%s,%s)' % (word1, word2, word3)] = ((word1, word2, word3) in message_words)
    return features

def extract_featuresUniBiTrigrams(message):
    message_words = set(message)
    features = {}
    for word1 in featureListUnigrams:
        features['contains(%s)' % (word1)] = ((word1) in message_words)
    for word1, word2 in featureListBigrams:
        features['contains(%s,%s)' % (word1, word2)] = ((word1, word2) in message_words)
    for word1, word2, word3 in featureListTrigrams:
        features['contains(%s,%s,%s)' % (word1, word2, word3)] = ((word1, word2, word3) in message_words)
    return features

#random.shuffle(pairs)

def execution(title, percent, extractor, pairs):
    print("\n\n" + title + " feature extration")
    n = int(len(pairs) * percent)
    train_set = nltk.classify.util.apply_features(extractor, pairs[:n])
    test_set = nltk.classify.util.apply_features(extractor, pairs[n:])
    print('train on %d instances, test on %d instances' % (n, len(pairs) - n))
    table = []
    table.append(assess_classifier(nltk.NaiveBayesClassifier.train(train_set), test_set, "Naive Bayes"))
    #table.append(assess_classifier(nltk.DecisionTreeClassifier.train(train_set), test_set, "Decision Tree"))
    #table.append(assess_classifier(nltk.MaxentClassifier.train(train_set, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10), test_set, "Maximum Entropy"))
    #table.append(assess_classifier(nltk.NaiveBayesClassifier.train(train_set), test_set, "Support Vector Machines"))
    print(tabulate(table, headers=["Classifier", "Accuracy", "Precision(pos)", "Recall(pos)", "F-measure(pos)",
                               "Precision(neg)", "Recall(neg)", "F-measure(neg)"]))
execution("BAG OF WORDS (100 Words)", percent, extract_featuresBagOfWords, pairsBagOfWords)
execution("BAG OF WORDS (More than 1 repetition)", percent, extract_featuresBagOfWordsRep, pairsBagOfWords)
execution("UNIGRAMS", percent, extract_featuresUnigrams, pairsUnigrams)
execution("BIGRAMS", percent, extract_featuresBigrams, pairsBigrams)
execution("TRIGRAMS", percent, extract_featuresTrigrams, pairsTrigrams)
list = []
for x in range(0, len(pairsUnigrams)):
    element = (pairsUnigrams[x][0] + pairsBigrams[x][0], pairsUnigrams[x][1])
    list.append(element)
execution("UNIGRAMS AND BIGRAMS COMBINED", percent, extract_featuresUniBigrams, list)
list = []
for x in range(0, len(pairsUnigrams)):
    element = (pairsBigrams[x][0] + pairsTrigrams[x][0], pairsUnigrams[x][1])
    list.append(element)
execution("BIGRAMS AND TRIGRAMS COMBINED", percent, extract_featuresBiTrigrams, list)
list = []
for x in range(0, len(pairsUnigrams)):
    element = (pairsUnigrams[x][0] + pairsBigrams[x][0] + pairsTrigrams[x][0], pairsUnigrams[x][1])
    list.append(element)
execution("UNIGRAMS, BIGRAMS AND TRIGRAMS COMBINED", percent, extract_featuresUniBiTrigrams, list)
#execution("TF-IDF feature extraction")

'''
#test the classifier
testMessage = 'Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis'
response = requests.get(baseUrl + '/nlp/processMessage/' + testMessage)
cleanTestMessage = ''
if(response.ok):
    cleanTestMessage = json.loads(response.content.decode('utf-8'))
else:
    response.raise_for_status()

response = requests.get(baseUrl + '/nlp/unigrams/' + cleanTestMessage)
if(response.ok):
    unigrams = json.loads(response.content.decode('utf-8'))

    print(NBClassifier.classify(extract_features(unigrams)))
    #print(NBClassifier.show_most_informative_features(10))
else:
    response.raise_for_status()
'''
