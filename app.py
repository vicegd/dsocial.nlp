#!flask/bin/python
from flask import Flask, jsonify, abort, make_response, request, url_for
#from flask.ext.httpauth import HTTPBasicAuth
from app_api.controllers.taskController import taskController
from app_api.controllers.movies.sentiment_movies_controller import sentiment_movies_controller

'''
auth = HTTPBasicAuth()
@auth.get_password
def get_password(username):
    if username == 'miguel':
        return 'python'
    return None

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)
'''

app = Flask(__name__)
app.register_blueprint(taskController)
app.register_blueprint(sentiment_movies_controller)

if __name__ == '__main__':
    app.run(debug=False)