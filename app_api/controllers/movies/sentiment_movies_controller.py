import pickle
from lib.feature_vector import FeatureVector
from lib.extract_features import ExtractFeatures
from adjustments.sentiment_movies_adjustment import SentimentMoviesAdjust
from flask import Blueprint
import json
import requests
from flask import Flask, jsonify, abort, make_response, request, url_for
from flask.ext.httpauth import HTTPBasicAuth

sentiment_movies_controller = Blueprint('sentiment', __name__, template_folder='templates')
@sentiment_movies_controller.route('/api/sentiment/movies/<text>', methods=['GET'])
def get_sentiment(text):
    f_classifier = open('./serialized/movies_sentiment_unigrams_bayes_classifier.pickle', 'rb')
    classifier = pickle.load(f_classifier)
    f_classifier.close()

    f_feature_list = open('./serialized/movies_sentiment_unigrams_feature_list.pickle', 'rb')
    feature_list = pickle.load(f_feature_list)
    f_feature_list.close()

    feature_vector = FeatureVector()
    processed_message = feature_vector.process_message(text)
    feature_vector = feature_vector.get_feature_vector_unigrams(processed_message)

    extract_features = ExtractFeatures(feature_list, None, None)
    features = extract_features.extract_features_unigrams(feature_vector)

    print(feature_vector)
    print(features)

    dist = classifier.prob_classify(features)
    adjustment = SentimentMoviesAdjust(dist.prob('negative'), dist.prob('positive'))
    response = adjustment.adjust(text)
    response['previousNegative'] = dist.prob('negative')
    response['previousPositive'] = dist.prob('positive')

    #print(classifier.show_most_informative_features(10))
    return jsonify(response)

@sentiment_movies_controller.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)

@sentiment_movies_controller.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)