#cross validate 2.0

import xml.etree.ElementTree as ET
import vectorize_bc3
import copy
from helper_function_variable import *

def vect_sent_thread(bc3_vector_dict, thread_listno):
	vector = [bc3_vector_dict[thread_listno][sent_ID] for sent_ID in bc3_vector_dict[thread_listno].keys()]

	return vector

def score_sent_thread(bc3_score_dict, thread_listno):
	score = [bc3_score_dict[thread_listno][sent_ID] for sent_ID in bc3_score_dict[thread_listno].keys()]

	return score

def validation_and_training_set_thread_listno(validation_first_thread_position, thread_listno_bc3):
	#validation_first_thread_position is the position of the first thread of the validation set in the original collections of thread
	validation_set_thread_listno = [thread_listno for thread_listno in thread_listno_bc3[validation_first_thread_position-1:validation_first_thread_position+4]]

	training_set_thread_listno = [thread_listno for thread_listno in  thread_listno_bc3 if thread_listno not in validation_set_thread_listno]

	return training_set_thread_listno, validation_set_thread_listno

def create_validation_and_training_set(bc3_vector_dict, bc3_score_dict, training_set_thread_listno, validation_set_thread_listno):

	validation_set_vector = []
	validation_set_score  = []
	training_set_vector = []
	training_set_score = []

	for thread_listno in validation_set_thread_listno:
		validation_set_vector += vect_sent_thread(bc3_vector_dict, thread_listno)
		validation_set_score  += score_sent_thread(bc3_score_dict, thread_listno)

	for thread_listno in training_set_thread_listno:
		training_set_vector += vect_sent_thread(bc3_vector_dict, thread_listno)
		training_set_score  += score_sent_thread(bc3_score_dict, thread_listno)

	return training_set_vector, training_set_score, validation_set_vector, validation_set_score

def validation(training_set_vector, training_set_score, validation_set_vector, validation_set_score):
	#Train classifier on traing_set_vector
	#Predict score for each sentence in validation_set_vector
	random_forest = vectorize_bc3.server_train_classifier(training_set_vector, training_set_score)

	validation_set_predicted_score = [] 

	for sentence in validation_set_vector:
		validation_set_predicted_score.append(float(random_forest.predict(sentence)))

	return validation_set_predicted_score

def update_predicted_score_each_sent(predicted_score_each_sent, validation_set_predicted_score, validation_set_thread_listno): 
    #Update predicted_score_each_sent with validation_set_predicted_score 

    for thread_listno in validation_set_thread_listno:
        sentence_count = 0
        for sent_ID in predicted_score_each_sent[thread_listno].keys():
            predicted_score_each_sent[thread_listno][sent_ID] = validation_set_predicted_score[sentence_count]
            sentence_count += 1
            if sentence_count == num_sent_each_thread[thread_listno]: break

def cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict):

    #predicted_score_each_sent contains the predicted score of each sentence after 10 fold cross validation
    predicted_score_each_sent = copy.deepcopy(bc3_vector_dict)

    #get all thread listno in bc3 
    thread_listno_bc3 = [thread_listno_of_current_thread(thread) for thread in root_corpus]

    for i in range(1,40,4):
        training_set_thread_listno, validation_set_thread_listno = validation_and_training_set_thread_listno(i, thread_listno_bc3)

        training_set_vector, training_set_score, validation_set_vector, validation_set_score = create_validation_and_training_set(bc3_vector_dict, bc3_score_dict, training_set_thread_listno, validation_set_thread_listno)

        #validation_set_predicted_score is the predicted score of the sentence in the validation set
        validation_set_predicted_score = validation(training_set_vector, training_set_score, validation_set_vector, validation_set_score)

        #update_predicted_score_each_sent updates the predicted_score_each_sent with the predicted score of the sentence in a validation set
        update_predicted_score_each_sent(predicted_score_each_sent, validation_set_predicted_score, validation_set_thread_listno)

    return predicted_score_each_sent

tree  = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root_corpus  = tree.getroot() 

predicted_score_each_sent = cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict)

test = []
test2 = []
for i in range(10):
    predicted_score_each_sent = cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict)

    test.append(predicted_score_each_sent["066-15270802"]["1.4"])
    test2.append(predicted_score_each_sent["066-15270802"]["1.3"])

print test
print test2
