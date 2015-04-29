#note: must note that sentences of equal importance might be arbitrarily included or excluded from the ideal summary because of sorting and length limit

import operator

from handy_function import *

#global variable
	#summary (type:dict) with the format
	#[thread_listno] = {} (sub dict)
	#sub dict contains the sentence_ID of ideal summary of <thread> that has <thread_listno>

def sort_dict_by_value(dict): 
    #Sorted a dictionary by values in descending order and turn into a tuple
    dict_sorted = sorted(dict.items(), key=operator.itemgetter(1), reverse = True)

    return dict_sorted

def return_ideal_summary_with_length_limit(length_limit, sentences_sorted_by_importance, thread_listno):
	#Return ideal summary of a <thread> with length limit of 30% of no. of words in original email 

	ideal_summary_word_count = 0

	ideal_summary_for_thread = []

	for sentence in sentences_sorted_by_importance:
		ideal_summary_for_thread.append(sentence[0])
		ideal_summary_word_count += ideal_summary_word_count + num_word_in_each_sentence[thread_listno][sentence[0]]
		if ideal_summary_word_count >= length_limit:
			return ideal_summary_for_thread


def return_ideal_summary(ideal_summary, thread_listno):
    
    #length_limit is the length limit of ideal summary - 30% of original email by word count 
    length_limit = sum(num_word_in_each_sentence[thread_listno].values())/100*30

    #Sort each sentence in a <thread> according to annotation score
    sentences_sorted_by_importance = sort_dict_by_value(annotation_score_each_sentence[thread_listno])

    ideal_summary_for_thread = return_ideal_summary_with_length_limit(length_limit, sentences_sorted_by_importance, thread_listno)

    ideal_summary_for_thread = sorted(ideal_summary_for_thread)        

    ideal_summary[thread_listno] = ideal_summary_for_thread

if __name__ == '__main__': 
    
    #loading necessary data from pickled files
    f = open("data/num_word_in_each_sentence.pck", "r")
    num_word_in_each_sentence = pickle.load(f)
    f.close()

    f = open("data/annotation_score_each_sentence.pck", "r")
    annotation_score_each_sentence = pickle.load(f)
    f.close()
    #done loading necessary data

    ideal_summary = dict()

    #update ideal_summary with the ideal summary of each <thread>
    for key in annotation_score_each_sentence:
        ideal_summary[key] = {}
        return_ideal_summary(ideal_summary, key)

    pickle_data(ideal_summary, "ideal_summary")