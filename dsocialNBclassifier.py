import nltk
import requests
import json

baseUrl = "http://localhost:4000/api"

messages = []
featureList = []
pairs = []
response = requests.get(baseUrl + '/messages')
if(response.ok):
    messages = json.loads(response.content.decode('utf-8'))
    for message in messages:
        featureList.extend(message['nlp']['unigrams'])
        pairs.append((message['nlp']['unigrams'], message['nlp']['sentiment']))
else:
    response.raise_for_status()

#remove featureList duplicates
featureList = list(set(featureList))

#start extract_features
def extract_features(msg):
    words = set(msg)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in words)
    return features

#extract feature vector for all tweets in one shote
n = int(len(pairs)*0.8)
training_set = nltk.classify.util.apply_features(extract_features, pairs[:n])
test_set = nltk.classify.util.apply_features(extract_features, pairs[n:])

# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print(nltk.classify.accuracy(NBClassifier, test_set))
print(nltk.metrics.precision(NBClassifier, test_set))
print(nltk.metrics.recall(NBClassifier, test_set))
print(nltk.metrics.f_measure(NBClassifier, test_set))






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
