import json
import nltk
import requests
from lib.feature_vector import FeatureVector
from lib.feature_list import FeatureList
from lib.extract_features import ExtractFeatures
from lib.train_classifier import TrainClassifier
from tabulate import tabulate

baseUrl = "http://localhost:4001/api"

data = []

pairsBagOfWords = []
featureListBagOfWords = []

featureListBagOfWordsRep = []

pairsUnigrams = []
pairsBigrams = []
pairsTrigrams = []

feature_list_unigrams = FeatureList('movies_sentiment_unigrams_feature_list')
feature_list_bigrams = FeatureList('movies_sentiment_bigrams_feature_list')
feature_list_trigrams = FeatureList('movies_sentiment_trigrams_feature_list')

feature_vector = FeatureVector()
train_classifier = TrainClassifier(10, 0.85)

response = requests.get(baseUrl + '/messages')
if(response.ok):
    data = json.loads(response.content.decode('utf-8'))
    for message in data['messages']:
        if(message['nlp']['sentiment'] != 'unknown'):
            processed_message = feature_vector.process_message(message['nlp']['source'])

            featureVectorBagOfWords = feature_vector.get_feature_vector_unigrams(processed_message)
            featureListBagOfWords.extend(featureVectorBagOfWords)
            pairsBagOfWords.append((featureVectorBagOfWords, message['nlp']['sentiment']))
 
            feature_vector_unigrams = feature_vector.get_feature_vector_unigrams(processed_message)
            feature_list_unigrams.extend(feature_vector_unigrams)
            pairsUnigrams.append((feature_vector_unigrams, message['nlp']['sentiment']))

            feature_vector_bigrams = feature_vector.get_feature_vector_bigrams(processed_message)
            feature_list_bigrams.extend(feature_vector_bigrams)
            pairsBigrams.append((feature_vector_bigrams, message['nlp']['sentiment']))

            feature_vector_trigrams = feature_vector.get_feature_vector_trigrams(processed_message)
            feature_list_trigrams.extend(feature_vector_trigrams)
            pairsTrigrams.append((feature_vector_trigrams, message['nlp']['sentiment']))

else:
    response.raise_for_status()

freqBagOfWords = nltk.FreqDist(featureListBagOfWords)
featureListBagOfWords = []
for element in freqBagOfWords.most_common(100):
    featureListBagOfWords.append(element[0])

for element in freqBagOfWords.most_common():
    if (element[1] > 1):
        featureListBagOfWordsRep.append(element[0])

feature_list_unigrams.to_set()
feature_list_bigrams.to_set()
feature_list_trigrams.to_set()

extract_features = ExtractFeatures(feature_list_unigrams.elements, feature_list_bigrams.elements, feature_list_trigrams.elements)

def execution(title, extractor, pairs):
    print("\n\n" + title + " feature extration")
    table = []
    table.append(train_classifier.train(nltk.NaiveBayesClassifier, extractor, pairs, "movies_sentiment_unigrams_bayes_classifier", "Naive Bayes"))
    #table.append(assess_classifier(nltk.DecisionTreeClassifier.train(train_set), test_set, "Decision Tree"))
    #table.append(assess_classifier(nltk.MaxentClassifier.train(train_set, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10), test_set, "Maximum Entropy"))
    #table.append(assess_classifier(nltk.NaiveBayesClassifier.train(train_set), test_set, "Support Vector Machines"))
    print(tabulate(table, headers=["Classifier", "Accuracy", "Precision(pos)", "Recall(pos)", "F-measure(pos)",
                               "Precision(neg)", "Recall(neg)", "F-measure(neg)"]))
#execution("BAG OF WORDS (100 Words)", extract_featuresBagOfWords, pairsBagOfWords)
#execution("BAG OF WORDS (More than 1 repetition)", extract_featuresBagOfWordsRep, pairsBagOfWords)
execution("UNIGRAMS", extract_features.extract_features_unigrams, pairsUnigrams)
#execution("BIGRAMS", extract_features.extract_features_bigrams, pairsBigrams)
#execution("TRIGRAMS", extract_features.extract_features_trigrams, pairsTrigrams)
#list = []
#for x in range(0, len(pairsUnigrams)):
#    element = (pairsUnigrams[x][0] + pairsBigrams[x][0], pairsUnigrams[x][1])
#    list.append(element)
#execution("UNIGRAMS AND BIGRAMS COMBINED", extract_featuresUniBigrams, list)
#list = []
#for x in range(0, len(pairsUnigrams)):
#    element = (pairsBigrams[x][0] + pairsTrigrams[x][0], pairsUnigrams[x][1])
#    list.append(element)
#execution("BIGRAMS AND TRIGRAMS COMBINED", extract_featuresBiTrigrams, list)
#list = []
#for x in range(0, len(pairsUnigrams)):
#    element = (pairsUnigrams[x][0] + pairsBigrams[x][0] + pairsTrigrams[x][0], pairsUnigrams[x][1])
#    list.append(element)
#execution("UNIGRAMS, BIGRAMS AND TRIGRAMS COMBINED", extract_featuresUniBiTrigrams, list)
#execution("TF-IDF feature extraction")

for message in data['messages']:
    if(message['nlp']['sentiment'] != 'unknown'):
        response = requests.get("http://localhost:5000/api/sentiment/movies/" + message['nlp']['source'])
        if(response.ok):
            data = json.loads(response.content.decode('utf-8'))
            if (data['previousPositive'] >= data['previousNegative']):
                previousSentiment = 'positive'
            else:
                previousSentiment = 'negative'
            if (data['positive'] >= data['negative']):
                sentiment = 'positive'
            else:
                sentiment = 'negative'
            if (message['nlp']['sentiment'] != previousSentiment):
                print('{0},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5},{6},{7}'.format(message['nlp']['source'],
                                           data['previousNegative'], data['previousPositive'],
                                           data['negative'], data['positive'],
                                           message['nlp']['sentiment'], previousSentiment, sentiment
                    ));
        #else:
        #    response.raise_for_status()