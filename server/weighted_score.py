from helpers import *
from cross_validate import *
import copy

# We measure the recall score against an ideal summary weighted across all annotators
# Recall score measures the percentages of importance sentences that were chosen

def recall_score_thread(ideal_summary, ideal_summary_from_cross_validation, thread_listno):
    '''Return the recall score for the <thread> that has thread_listno.'''
    recall_thread = 0.0
    for sent_ID in ideal_summary[thread_listno]:
        if sent_ID in ideal_summary_from_cross_validation[thread_listno]:
            recall_thread += 1

    recall_thread = recall_thread / len(ideal_summary[thread_listno])
    return recall_thread

if __name__ == '__main__':
    ideal_summary_from_cross_validation = copy.deepcopy(predicted_score_each_sent)

    for thread_listno in predicted_score_each_sent:
        ideal_summary_from_cross_validation[thread_listno] = ideal_summary_thread(thread_listno, predicted_score_each_sent, num_word_each_sent)

    recall = []
    for thread_listno in ideal_summary:
        recall_thread = recall_score_thread(ideal_summary, ideal_summary_from_cross_validation, thread_listno)
        recall.append((thread_listno, recall_thread))

    print(recall)
