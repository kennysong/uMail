#Return weighted score 

from helper_function_variable import *
import copy

#we measure the recall score against an ideal summary weighted across all annotators 
#Recall score measures the percentages of importance sentences that were chosen 

if __name__ == '__main__':
    #Load pickled files
    f = open("data/num_word_in_each_sentence.pck", "r")
    num_word_in_each_sentence = pickle.load(f)
    f.close()

    f = open("data/predicted_score_each_sentence.pck", "r")
    predicted_score_each_sentence = pickle.load(f)
    f.close()

    f = open("data/ideal_summary.pck", "r")
    ideal_summary = pickle.load(f)
    f.close()
    #finish loading

    ideal_summary_from_cross_validation = copy.deepcopy(predicted_score_each_sentence)
    
    for key in predicted_score_each_sentence:
        ideal_summary_from_cross_validation[key] = ideal_summary_thread(key, predicted_score_each_sentence, num_word_in_each_sentence)

    print ideal_summary_from_cross_validation