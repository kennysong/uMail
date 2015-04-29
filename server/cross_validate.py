#cross validate 2.0

import xml.etree.ElementTree as ET
import vectorize_bc3
import copy
from handy_function import *

def vector_of_sentence_of_thread_from_bc3_vector_dict(thread_listno):
	vector = [bc3_vector_dict[thread_listno][sentence_ID] for sentence_ID in bc3_vector_dict[thread_listno].keys()]

	return vector

def score_of_sentence_of_thread_from_bc3_score_dict(thread_listno):
	score = [bc3_score_dict[thread_listno][sentence_ID] for sentence_ID in bc3_vector_dict[thread_listno].keys()]

	return score

def validation_and_training_set_thread_listno(validation_first_thread_position):
	validation_set_thread_listno = [thread_listno for thread_listno in thread_listno_in_bc3[validation_first_thread_position-1:validation_first_thread_position+4]]

	training_set_thread_listno = [thread_listno for thread_listno in  thread_listno_in_bc3 if thread_listno not in validation_set_thread_listno]

	return training_set_thread_listno, validation_set_thread_listno

def create_validation_and_training_set(training_set_thread_listno, validation_set_thread_listno):

	validation_set_vector = []
	validation_set_score  = []
	training_set_vector = []
	training_set_score = []

	for thread_listno in validation_set_thread_listno:
		validation_set_vector += vector_of_sentence_of_thread_from_bc3_vector_dict(thread_listno)
		validation_set_score  += score_of_sentence_of_thread_from_bc3_score_dict(thread_listno)

	for thread_listno in training_set_thread_listno:
		training_set_vector += vector_of_sentence_of_thread_from_bc3_vector_dict(thread_listno)
		training_set_score  += score_of_sentence_of_thread_from_bc3_score_dict(thread_listno)

	return training_set_vector, training_set_score, validation_set_vector, validation_set_score

def update_predicted_score_each_sentence_with_validation_set_predicted_score(predicted_score_each_sentence, validation_set_predicted_score, validation_set_thread_listno): 

	for thread_listno in validation_set_thread_listno:
		sentence_count = 0
		for sentence_ID in predicted_score_each_sentence[thread_listno].keys():
			predicted_score_each_sentence[thread_listno][sentence_ID] = validation_set_predicted_score[sentence_count]
			sentence_count += 1
			if sentence_count == num_sentence_in_each_thread[thread_listno]: break

def validation(training_set_vector, training_set_score, validation_set_vector, validation_set_score):
	#Train classifier on traing_set_vector
	#Predict score for each sentence in validation_set_vector
	random_forest = vectorize_bc3.server_train_classifier(training_set_vector,training_set_score)

	validation_set_predicted_score = [] 

	for sentence in validation_set_vector:
		validation_set_predicted_score.append(float(random_forest.predict(sentence)))

	return validation_set_predicted_score

if __name__ == '__main__':

	#Load necessary data 
	f = open("data/bc3_vector_dict.pck", "r")
	bc3_vector_dict = pickle.load(f)
	f.close()

	f = open("data/bc3_score_dict.pck", "r")
	bc3_score_dict = pickle.load(f)
	f.close()

	f = open("data/num_sentence_in_each_thread.pck", "r")
	num_sentence_in_each_thread = pickle.load(f)
	f.close()

	tree  = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
	root  = tree.getroot() 
	#Finish loading

	predicted_score_each_sentence = copy.deepcopy(bc3_vector_dict)

	#get all thread listno in bc3 
	thread_listno_in_bc3 = [thread_listno_of_current_thread(thread) for thread in root]

	#create validation set and training set
	for i in range(1,40,4):
		training_set_thread_listno, validation_set_thread_listno = validation_and_training_set_thread_listno(i)

		training_set_vector, training_set_score, validation_set_vector, validation_set_score = create_validation_and_training_set(training_set_thread_listno, validation_set_thread_listno)

		validation_set_predicted_score = validation(training_set_vector, training_set_score, validation_set_vector, validation_set_score)

		update_predicted_score_each_sentence_with_validation_set_predicted_score(predicted_score_each_sentence, validation_set_predicted_score, validation_set_thread_listno)

	print predicted_score_each_sentence

