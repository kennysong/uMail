import xml.etree.ElementTree as ET
import vectorize_bc3
import copy
from helpers import *

def recall_score_thread(ideal_summary, predicted_summary, thread_listno):
    '''Return the recall score for the <thread> that has thread_listno.'''
    recall_thread = 0.0
    for sent_ID in ideal_summary[thread_listno]:
        if sent_ID in predicted_summary[thread_listno]:
            recall_thread += 1

    recall_thread = recall_thread / len(ideal_summary[thread_listno])
    return recall_thread

def vect_sent_thread(bc3_vector_dict, thread_listno):
    '''Return the vectors of the sentences  of the <thread> that has thread_listno.'''

    vectors = [bc3_vector_dict[thread_listno][sent_ID] for sent_ID in bc3_vector_dict[thread_listno].keys()]
    return vectors

def score_sent_thread(bc3_score_dict, thread_listno):
    '''Return the scores of the sentences of the <thread> that has thread_listno.'''

    scores = [bc3_score_dict[thread_listno][sent_ID] for sent_ID in bc3_score_dict[thread_listno].keys()]
    return scores

def split_valid_train_listnos(validation_first_thread_position, thread_listnos):
    '''Return the thread_listno of the threads in training set and validation set.
    validation_first_thread_position is the position in the original collections of thread of the first thread of the validation set.'''

    valid_listnos = [thread_listno for thread_listno in thread_listnos[validation_first_thread_position-1:validation_first_thread_position+4]]
    train_listnos = [thread_listno for thread_listno in  thread_listnos if thread_listno not in valid_listnos]
    return train_listnos, valid_listnos

def create_valid_train_set(bc3_vector_dict, bc3_score_dict, train_listnos, valid_listnos):
    '''Return training, validation set and scores from the thread_listno of the threads in training set and validation set.'''

    valid_vectors = []
    valid_scores  = []
    train_vectors = []
    train_scores = []

    for thread_listno in valid_listnos:
        valid_vectors += vect_sent_thread(bc3_vector_dict, thread_listno)
        valid_scores  += score_sent_thread(bc3_score_dict, thread_listno)

    for thread_listno in train_listnos:
        train_vectors += vect_sent_thread(bc3_vector_dict, thread_listno)
        train_scores  += score_sent_thread(bc3_score_dict, thread_listno)

    return train_vectors, train_scores, valid_vectors, valid_scores

def update_predicted_sent_scores(predicted_sent_scores, valid_predicted_scores, valid_listnos, num_sent_each_thread): 
    '''Update predicted_sent_scores with valid_predicted_scores after every validation run.'''

    for thread_listno in valid_listnos:
        sentence_count = 0
        for sent_ID in predicted_sent_scores[thread_listno].keys():
            predicted_sent_scores[thread_listno][sent_ID] = valid_predicted_scores[sentence_count]
            sentence_count += 1
            if sentence_count == num_sent_each_thread[thread_listno]: break

def cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict, ideal_summaries, num_sent_each_thread, num_word_each_sent):
    '''Perform 10 fold cross validation. Will return the average weighted recall score for all the validation sets.'''

    # predicted_sent_scores contains the predicted score of each sentence after 10 fold cross validation
    # Copy the structure of bc3_vector_dict
    # {thread_listno: {sentence_id: score, sentence_id2: score, ...}, ...}
    predicted_sent_scores = copy.deepcopy(bc3_vector_dict)

    # Get all thread listno in bc3 
    thread_listnos = [thread_listno_of_current_thread(thread) for thread in root_corpus]

    # Run 10-fold cross validation, collecting the predicted score for each sentence in the corpus
    for i in range(1, 40, 4):
        # Split BC3 vectors into training and validation sets
        train_listnos, valid_listnos = split_valid_train_listnos(i, thread_listnos)
        train_vectors, train_scores, valid_vectors, valid_scores = \
            create_valid_train_set(bc3_vector_dict, bc3_score_dict, train_listnos, valid_listnos)

        # Train a classifier and test on validation set, recording the scores
        random_forest = vectorize_bc3.train_classifier(train_vectors, train_scores)
        valid_predicted_scores = [float(random_forest.predict(sentence)) for sentence in valid_vectors]

        # update_predicted_sent_scores updates the predicted_sent_scores with the predicted score of the sentence in a validation set
        update_predicted_sent_scores(predicted_sent_scores, valid_predicted_scores, valid_listnos, num_sent_each_thread)

   
    # Get the summaries for each thread in the same dictionary structure
    cross_validation_summaries = copy.deepcopy(predicted_sent_scores)
    for thread_listno in cross_validation_summaries:
        cross_validation_summaries[thread_listno] = get_thread_summary(thread_listno, predicted_sent_scores, num_word_each_sent)

    # Calculate the recall of each thread
    recalls = []
    for thread_listno in ideal_summaries:
        recall = recall_score_thread(ideal_summaries, cross_validation_summaries, thread_listno)
        recalls.append(recall)

    # Calculate the average recall over all threads
    avg_recall = sum(recalls) / len(recalls)

    return avg_recall

if __name__ == '__main__':
    # Set up some variables to calculate recall
    root_annotation = ET.parse("bc3_corpus/bc3corpus.1.0/annotation.xml").getroot()
    root_corpus = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml").getroot()

    # Import precalculated variables for bc3 corpus
    from data.helper_variables import bc3_vector_dict, bc3_score_dict, ideal_summaries
    from data.helper_variables import num_sent_each_thread, num_word_each_sent

    # Calculate and print out recall several times
    for i in range(100):
        recall = cross_validate(root_corpus, bc3_vector_dict, bc3_score_dict, 
                 ideal_summaries, num_sent_each_thread, num_word_each_sent)
        print(recall)
