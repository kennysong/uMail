import xml.etree.ElementTree as ET
import copy
from helpers import * 

## Entire file debugged by Kenny, Quan on May 8th ##

def weighted_recall(ideal_summary, predicted_summary, anno_sent_scores, thread_listno):
    '''Return the weighted recall score for the this summary.'''

    # A dictionary of {sentID: annotated_score, ...} for all sentences in a thread
    # Note: This does not include sentIDs with score 0! 
    thread_scores = anno_sent_scores[thread_listno]

    # Get the sum of the sentence scores in the predicted summary
    sum_score_of_predicted_summary = 0.0;
    for sent_ID in predicted_summary:
        sum_score_of_predicted_summary += thread_scores.get(sent_ID, 0)

    # Get the sum of the sentence scores in the ideal summary
    sum_score_of_ideal_summary = 0.0;
    for sent_ID in ideal_summary:
        sum_score_of_ideal_summary += thread_scores.get(sent_ID, 0)

    return sum_score_of_predicted_summary / sum_score_of_ideal_summary

def vectors_of_thread(bc3_vector_dict, thread_listno):
    '''Return the sentence vectors of the <thread> that has thread_listno.'''
    sent_IDs = sorted(bc3_vector_dict[thread_listno].keys())
    vectors = [bc3_vector_dict[thread_listno][sent_ID] for sent_ID in sent_IDs]
    return vectors

def scores_of_thread(bc3_score_dict, thread_listno):
    '''Return the scores of the sentences of the <thread> that has thread_listno.'''
    sent_IDs = sorted(bc3_score_dict[thread_listno].keys())
    scores = [bc3_score_dict[thread_listno][sent_ID] for sent_ID in sent_IDs]
    return scores

def get_valid_train_listnos(valid_start_index, thread_listnos, valid_size=4):    
    '''Return the thread_listno of the threads in training set and validation set.
       valid_start_index is the position of the first thread in the validation set.'''
    valid_listnos = [listno for listno in thread_listnos[valid_start_index: valid_start_index+valid_size]]
    train_listnos = [listno for listno in thread_listnos if listno not in valid_listnos]

    return train_listnos, valid_listnos

def make_valid_train_sets(bc3_vector_dict, bc3_score_dict, train_listnos, valid_listnos):
    '''Return training, validation set and scores from the thread_listno of the threads in train_set and valid_set.'''

    # Make validation set
    valid_vectors, valid_scores = [], []
    for thread_listno in valid_listnos:
        valid_vectors += vectors_of_thread(bc3_vector_dict, thread_listno)
        valid_scores  += scores_of_thread(bc3_score_dict, thread_listno)

    # Make training set
    train_vectors, train_scores = [], []
    for thread_listno in train_listnos:
        train_vectors += vectors_of_thread(bc3_vector_dict, thread_listno)
        train_scores  += scores_of_thread(bc3_score_dict, thread_listno)

    return train_vectors, train_scores, valid_vectors, valid_scores

def update_predicted_sent_scores(predicted_sent_scores, valid_predicted_scores, valid_listnos): 
    '''Update predicted_sent_scores with valid_predicted_scores after every validation run.'''

    for listno in valid_listnos:
        # Update sentences scores for this thread
        sent_IDs = sorted(predicted_sent_scores[listno].keys())
        for i in range(len(sent_IDs)):
            sent_ID = sent_IDs[i]
            predicted_sent_scores[listno][sent_ID] = valid_predicted_scores[i]

def cross_validate(bc3_vector_dict, bc3_score_dict, ideal_summaries, num_word_each_sent, anno_sent_scores):
    '''Perform 10 fold cross validation. Will return the average weighted recall score for all the validation sets.'''

    # A dictionary with structure: {thread_listno: {sentence_id: score, sentence_id2: score, ...}, ...}
    predicted_sent_scores = empty_bc3_structure_dict()

    # Get all thread listno in bc3
    corpus_root = ET.parse(os.path.dirname(os.path.abspath(__file__)) + "/bc3_corpus/corpus_processed.xml").getroot()
    thread_listnos = [listno_of_thread(thread) for thread in corpus_root]

    # Run 10-fold cross validation, collecting the predicted score for each sentence in the corpus
    for i in range(0, 40, 4):
        # Split BC3 vectors into training and validation sets
        train_listnos, valid_listnos = get_valid_train_listnos(i, thread_listnos)
        train_vectors, train_scores, valid_vectors, valid_scores = \
            make_valid_train_sets(bc3_vector_dict, bc3_score_dict, train_listnos, valid_listnos)

        # Train a classifier and test on validation set, recording the scores
        random_forest = train_classifier(train_vectors, train_scores)
        valid_predicted_scores = [float(random_forest.predict(sent)) for sent in valid_vectors]

        # update_predicted_sent_scores updates the dictionary in place with new predicted scores
        update_predicted_sent_scores(predicted_sent_scores, valid_predicted_scores, valid_listnos)

    # Get the summaries for each thread in the same dictionary structure 
    predicted_summaries = empty_bc3_structure_dict()
    for thread_listno in predicted_summaries:
        predicted_summaries[thread_listno] = get_thread_summary(thread_listno, predicted_sent_scores, num_word_each_sent)

    # Calculate the recall of each thread
    recalls = []
    for thread_listno in ideal_summaries:
        ideal_summary = ideal_summaries[thread_listno]
        predicted_summary = predicted_summaries[thread_listno]
        recall = weighted_recall(ideal_summary, predicted_summary, anno_sent_scores, thread_listno)

        recalls.append(recall)

    # Calculate the average recall over all threads
    avg_recall = sum(recalls) / len(recalls)

    return avg_recall

if __name__ == '__main__':
    # Import precalculated variables for bc3 corpus
    from data.helper_variables import bc3_vector_dict, bc3_score_dict, ideal_summaries
    from data.helper_variables import num_word_each_sent, anno_sent_scores

    # Calculate and print out recall several times
    for i in range(10):
        recall = cross_validate(bc3_vector_dict, bc3_score_dict, ideal_summaries,
                                num_word_each_sent, anno_sent_scores)
        print(recall)