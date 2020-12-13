from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import time
import cv2
import json
from service import Person
app = Flask(__name__)


@app.route('/hello', methods=['GET'])
def get():
    return 'hello i am dung'


@app.route('/create', methods=['POST'])
def create():
    if request.method == 'POST':
        person = Person()
        content = request.get_json()
        response = person.create(content['name'])
        return response


@app.route('/update', methods=['POST'])
def update():
    if request.method == 'POST':
        person = Person()
        value = request.form['name']
        id = request.form['id']
        person.update(id, value, request.files['img'])
    return 'Done'


@app.route('/recognize', methods=['GET'])
def recognize():
    
    return 'Done'


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
