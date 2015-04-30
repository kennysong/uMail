#Return weighted score 

from helper_function_variable import *
from cross_validate import *
import copy

#we measure the recall score against an ideal summary weighted across all annotators 
#Recall score measures the percentages of importance sentences that were chosen 

if __name__ == '__main__':

    ideal_summary_from_cross_validation = copy.deepcopy(predicted_score_each_sent)
    
    for thread_listno in predicted_score_each_sent:
        ideal_summary_from_cross_validation[thread_listno] = ideal_summary_thread(thread_listno, predicted_score_each_sent, num_word_each_sent)

    print ideal_summary_from_cross_validation