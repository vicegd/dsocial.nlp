import collections
import json
import nltk
import requests
import random
import math
import pickle
import lib.feature_vector as feature_vector
import lib.extract_features as extract_feactures
import lib.classifiers.extendedNaiveBayesClassifier as extendedNaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.metrics import precision, recall, f_measure
from tabulate import tabulate

baseUrl = "http://localhost:4001/api"

#negatives = twitter_samples.strings('negative_tweets.json')
#positives = twitter_samples.strings('positive_tweets.json')

def assess_classifier(classifier, extractor, pairs, text):
    #random.shuffle(pairs)
    kfolds = 10
    instancesPerKfold = math.ceil((len(pairs) / kfolds))
    percentToTrain = 0.85
    instancesPerKfoldToTrain = math.ceil(instancesPerKfold*percentToTrain)
    instancesPerKfoldToTest = instancesPerKfold-instancesPerKfoldToTrain

    print('Total number of instances = %s' % (len(pairs)))
    print('Number of K-folds = %s' % kfolds)
    print('Number of instances in each k-fold = %s' % instancesPerKfold)
    print('Percentage of instances trained in each k-fold = %s' % percentToTrain)
    print('Number of instances per k-fold to train = %s' % instancesPerKfoldToTrain)
    print('Number of instances per k-fold to test = %s' % instancesPerKfoldToTest)

    #k-folds
    totalAccuracy = 0
    for i in range(1, kfolds+1):
        j1 = (i-1)*instancesPerKfold
        if (i == kfolds):
            j2 = len(pairs)-instancesPerKfoldToTest
            k1 = j2
            k2 = len(pairs)
        else:
            j2 = j1 + instancesPerKfoldToTrain
            k1 = j2
            k2 = j2 + instancesPerKfoldToTest
        train_set = nltk.classify.util.apply_features(extractor, pairs[j1:j2])
        test_set = nltk.classify.util.apply_features(extractor, pairs[k1:k2])
        trained_classifier = classifier.train(train_set)
        f = open('serialized/movies_sentiment_unigrams_bayes_classifier.pickle', 'wb')
        pickle.dump(trained_classifier, f)
        f.close()
        accuracy = nltk.classify.accuracy(trained_classifier, test_set)
        print('\t K-fold %s training[%s,%s) testing[%s,%s) - accuracy:%s' % (i, j1, j2, k1, k2, accuracy))
        totalAccuracy += accuracy

    #individual
    instances = math.ceil(len(pairs))
    instancesToTrain = math.ceil(instances*percentToTrain)
    train_set = nltk.classify.util.apply_features(extractor, pairs[:instancesToTrain])
    test_set = nltk.classify.util.apply_features(extractor, pairs[instancesToTrain:instances])
    trained_classifier = classifier.train(train_set)
    accuracy = nltk.classify.accuracy(trained_classifier, test_set)
    print('\t Complete training[%s,%s) testing[%s,%s) - accuracy:%s' % (0, instancesToTrain, instancesToTrain, instances, accuracy))

    return [text, totalAccuracy/kfolds, 0, 0, 0, 0, 0, 0]
    '''
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
    '''

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
            processedMessage = feature_vector.processMessage(message['nlp']['source'])

            featureVectorBagOfWords = feature_vector.getFeatureVectorUnigrams(processedMessage)
            featureListBagOfWords.extend(featureVectorBagOfWords)
            pairsBagOfWords.append((featureVectorBagOfWords, message['nlp']['sentiment']))

            featureVectorUnigrams = feature_vector.getFeatureVectorUnigrams(processedMessage)
            featureListUnigrams.extend(featureVectorUnigrams)
            pairsUnigrams.append((featureVectorUnigrams, message['nlp']['sentiment']))

            featureVectorBigrams = feature_vector.getFeatureVectorBigrams(processedMessage)
            featureListBigrams.extend(featureVectorBigrams)
            pairsBigrams.append((featureVectorBigrams, message['nlp']['sentiment']))

            featureVectorTrigrams = feature_vector.getFeatureVectorTrigrams(processedMessage)
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
f = open('serialized/movies_sentiment_unigrams_feature_list.pickle', 'wb')
pickle.dump(featureListUnigrams, f)
f.close()

featureListBigrams = list(set(featureListBigrams))
featureListTrigrams = list(set(featureListTrigrams))

def execution(title, extractor, pairs):
    print("\n\n" + title + " feature extration")
    table = []
    table.append(assess_classifier(nltk.NaiveBayesClassifier, extractor, pairs, "Naive Bayes"))
    #table.append(assess_classifier(nltk.DecisionTreeClassifier.train(train_set), test_set, "Decision Tree"))
    #table.append(assess_classifier(nltk.MaxentClassifier.train(train_set, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10), test_set, "Maximum Entropy"))
    #table.append(assess_classifier(nltk.NaiveBayesClassifier.train(train_set), test_set, "Support Vector Machines"))
    print(tabulate(table, headers=["Classifier", "Accuracy", "Precision(pos)", "Recall(pos)", "F-measure(pos)",
                               "Precision(neg)", "Recall(neg)", "F-measure(neg)"]))
#execution("BAG OF WORDS (100 Words)", extract_featuresBagOfWords, pairsBagOfWords)
#execution("BAG OF WORDS (More than 1 repetition)", extract_featuresBagOfWordsRep, pairsBagOfWords)
execution("UNIGRAMS", extract_feactures.extract_features_unigrams(featureListUnigrams), pairsUnigrams)
#execution("BIGRAMS", extract_featuresBigrams, pairsBigrams)
#execution("TRIGRAMS", extract_featuresTrigrams, pairsTrigrams)
list = []
for x in range(0, len(pairsUnigrams)):
    element = (pairsUnigrams[x][0] + pairsBigrams[x][0], pairsUnigrams[x][1])
    list.append(element)
#execution("UNIGRAMS AND BIGRAMS COMBINED", extract_featuresUniBigrams, list)
list = []
for x in range(0, len(pairsUnigrams)):
    element = (pairsBigrams[x][0] + pairsTrigrams[x][0], pairsUnigrams[x][1])
    list.append(element)
#execution("BIGRAMS AND TRIGRAMS COMBINED", extract_featuresBiTrigrams, list)
list = []
for x in range(0, len(pairsUnigrams)):
    element = (pairsUnigrams[x][0] + pairsBigrams[x][0] + pairsTrigrams[x][0], pairsUnigrams[x][1])
    list.append(element)
#execution("UNIGRAMS, BIGRAMS AND TRIGRAMS COMBINED", extract_featuresUniBiTrigrams, list)
#execution("TF-IDF feature extraction")

text = "Hello this is a test. I liked the movie so much"
