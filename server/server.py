import os
import json
from flask import Flask
from flask import request

from vectorize_email import process_email

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def hello():
    return "Hello world!"

@app.route('/new_email', methods=['POST'])
def new_email():
    subject = request.form['subject']
    email_text = request.form['email']
    to = request.form['to']
    cc = request.form['cc']
    to_cc = to + ',' + cc

    sent_sorted, sent_index = process_email(email_text, subject, to_cc)
    summary = {'sent_sorted': sent_sorted, 'sent_index': sent_index}
    summaryJSON = json.dumps(summary)

    return summaryJSON

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
