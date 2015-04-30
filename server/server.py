import os
import vectorize_bc3
from flask import Flask

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def hello():
    return "Hello world!"
