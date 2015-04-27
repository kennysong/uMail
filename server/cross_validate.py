import xml.etree.ElementTree as ET

import vectorize_bc3
from data.vocab_data import *
from data.vectorized_bc3_data import *

#using the first 36 emails as training set and last 4 as validation set
tree  = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root  = tree.getroot() 

number_sentence_last_4_emails = 0 

#Calculating the number of sentences in the last 4 email threads
for i in range(1,5):
    for j in range(len(root[-i])):
        if root[-i][j].tag == "DOC":
            for k in range(len(root[-i][j])):
                if root[-i][j][k].tag == "Text":
                    print root[-i][j][k]
                    print number_sentence_last_4_emails
                    number_sentence_last_4_emails += len(root[-i][j][k])

training_set_vector = aligned_sentence_vectors[0:(3222-number_sentence_last_4_emails-1)]
training_set_score  = aligned_scores[0:(3222-number_sentence_last_4_emails-1)]

validation_set_vector = aligned_sentence_vectors[(3222-number_sentence_last_4_emails):3221]
validation_set_score  = aligned_scores[(3222-number_sentence_last_4_emails):3221]

#traing the classifier on the first 36 threads
random_forest = vectorize_bc3.server_train_classifier(training_set_vector,training_set_score)

validation_set_predicted_score = [] 

for i in range(len(validation_set_vector)):
    validation_set_predicted_score.append(random_forest.predict(validation_set_vector[i]))

for i in range(len(validation_set_predicted_score)):
    validation_set_predicted_score[i] = float(validation_set_predicted_score[i])

# the length of the summary is limited to 30% of the original length of the email 
difference = []
for i in range(len(validation_set_predicted_score)):
    difference.append(validation_set_predicted_score[i] - validation_set_score[i])
