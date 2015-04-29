#Create ideal summary 
import operator
from data.length_each_sentence_data import *
from data.annotation_score_each_sentence_data import *

def sort_dict_by_value(dict): 
    #Sorted the dictionary by values and turn into a tuple
    dict_sorted = sorted(dict.items(), key=operator.itemgetter(1), reverse = True)

    return dict_sorted

def return_ideal_summary(summary, thread_listno):
    
    #length_limit is the length limit of ideal summary - 30% of original email by word count 
    length_limit = sum(length_each_sentence[thread_listno].itervalues())/100*30

    sentences_sorted_by_importance = sort_dict_by_value(annotation_score_each_sentence[thread_listno])

    ideal_summary_word_count = 0

    ideal_summary = []
    for sentence in sentences_sorted_by_importance:
        ideal_summary.append(sentence[0])
        ideal_summary_word_count += ideal_summary_word_count + length_each_sentence[thread_listno][sentence[0]]
        if ideal_summary_word_count >= length_limit:
            break

    ideal_summary = sorted(ideal_summary)        

    summary[thread_listno] = ideal_summary

if __name__ == '__main__': 
    summary = dict()
    for key in annotation_score_each_sentence:
        summary[key] = {}
        return_ideal_summary(summary, key)
    print summary