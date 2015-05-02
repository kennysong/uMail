import os
import vectorize_bc3
from flask import Flask
from flask import request

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def hello():
    return "Hello world!"

@app.route('/new_email', methods=['POST'])
def new_email():
    return request.form['subject']

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)