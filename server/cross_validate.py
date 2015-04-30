import xml.etree.ElementTree as ET
import vectorize_bc3
import copy
from helpers import *

def vect_sent_thread(bc3_vector_dict, thread_listno):
    '''Return the vectors of the sentences  of the <thread> that has thread_listno.'''

    vectors = [bc3_vector_dict[thread_listno][sent_ID] for sent_ID in bc3_vector_dict[thread_listno].keys()]
    return vectors

def score_sent_thread(bc3_score_dict, thread_listno):
    '''Return the scores of the sentences of the <thread> that has thread_listno.'''

    scores = [bc3_score_dict[thread_listno][sent_ID] for sent_ID in bc3_score_dict[thread_listno].keys()]
    return scores

def valid_train_set_thread_listno(validation_first_thread_position, thread_listno_bc3):
    '''Return the thread_listno of the threads in training set and validation set.
    validation_first_thread_position is the position in the original collections of thread of the first thread of the validation set.'''

    valid_set_thread_listno = [thread_listno for thread_listno in thread_listno_bc3[validation_first_thread_position-1:validation_first_thread_position+4]]
    train_set_thread_listno = [thread_listno for thread_listno in  thread_listno_bc3 if thread_listno not in valid_set_thread_listno]
    return train_set_thread_listno, valid_set_thread_listno

def create_valid_train_set(bc3_vector_dict, bc3_score_dict, train_set_thread_listno, valid_set_thread_listno):
    '''Return training, validation set and scores from the thread_listno of the threads in training set and validation set.'''

    valid_set_v = []
    validation_set_score  = []
    train_set_v = []
    training_set_score = []

    for thread_listno in valid_set_thread_listno:
        valid_set_v += vect_sent_thread(bc3_vector_dict, thread_listno)
        validation_set_score  += score_sent_thread(bc3_score_dict, thread_listno)

    for thread_listno in train_set_thread_listno:
        train_set_v += vect_sent_thread(bc3_vector_dict, thread_listno)
        training_set_score  += score_sent_thread(bc3_score_dict, thread_listno)

    return train_set_v, training_set_score, valid_set_v, validation_set_score

def validation(train_set_v, training_set_score, valid_set_v, validation_set_score):
    '''Train classifier on train_set_v. Predict score for each sentence in valid_set_v.'''

    random_forest = vectorize_bc3.train_full_classifier(train_set_v, training_set_score)
    validation_set_predicted_score = [] 

    for sentence in valid_set_v:
        validation_set_predicted_score.append(float(random_forest.predict(sentence)))

    return validation_set_predicted_score

def update_predicted_score_each_sent(predicted_score_each_sent, validation_set_predicted_score, valid_set_thread_listno): 
    '''Update predicted_score_each_sent with validation_set_predicted_score after every validation run.'''

    for thread_listno in valid_set_thread_listno:
        sentence_count = 0
        for sent_ID in predicted_score_each_sent[thread_listno].keys():
            predicted_score_each_sent[thread_listno][sent_ID] = validation_set_predicted_score[sentence_count]
            sentence_count += 1
            if sentence_count == num_sent_each_thread[thread_listno]: break

def cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict):
    '''Perform 10 fold cross validation. Returns what?'''

    # predicted_score_each_sent contains the predicted score of each sentence after 10 fold cross validation
    # Copy the structure of bc3_vector_dict
    predicted_score_each_sent = copy.deepcopy(bc3_vector_dict)

    # Get all thread listno in bc3 
    thread_listno_bc3 = [thread_listno_of_current_thread(thread) for thread in root_corpus]

    # Run cross-validation woohoo
    for i in range(1,40,4):
        train_set_thread_listno, valid_set_thread_listno = valid_train_set_thread_listno(i, thread_listno_bc3)

        train_set_v, training_set_score, valid_set_v, validation_set_score = create_valid_train_set(bc3_vector_dict, bc3_score_dict, train_set_thread_listno, valid_set_thread_listno)

        # validation_set_predicted_score is the predicted score of the sentence in the validation set
        validation_set_predicted_score = validation(train_set_v, training_set_score, valid_set_v, validation_set_score)

        # update_predicted_score_each_sent updates the predicted_score_each_sent with the predicted score of the sentence in a validation set
        update_predicted_score_each_sent(predicted_score_each_sent, validation_set_predicted_score, valid_set_thread_listno)

    return predicted_score_each_sent

tree  = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root_corpus  = tree.getroot() 

predicted_score_each_sent = cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict)

# See the variance in predicted score for each sentence by uncomment and run the section below
# test = []
# test2 = []
# for i in range(20):
#     predicted_score_each_sent = cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict)

#     test.append(predicted_score_each_sent["066-15270802"]["1.4"])
#     test2.append(predicted_score_each_sent["066-15270802"]["1.3"])

# print test
# print 
# print test2
