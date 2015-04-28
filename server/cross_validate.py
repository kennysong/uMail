import xml.etree.ElementTree as ET

import vectorize_bc3
from data.vocab_data import *
from data.vectorized_bc3_data import *

#using the first 36 emails as training set and last 4 as validation set
tree  = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root  = tree.getroot() 

number_sentence_last_4_emails = 0 

#sequence of root: root -> thread -> DOC -> text -> sent (each sent being one sentence in an email)
# the length of the summary is limited to 30% of  the original length of the email 

#definition of weighted recall:
#The recall score measures the percentage of important sentences that were chosen
#In weighted recall, the Gold Standard summary is the highest scoring sentences up to a summary length of 30% 

def cal_num_sentence_in_emails(starting_email_thread, ending_email_thread):
    #return total number of sentences from starting_email_thread to ending_email_thread, including sentences from both of them

    No_sentence = 0
    for i in range(starting_email_thread-1,ending_email_thread):
        for j in range(len(root[i])):
            if root[i][j].tag == "DOC":
                for k in range(len(root[i][j])):
                    if root[i][j][k].tag == "Text":
                        No_sentence += len(root[i][j][k])

    return No_sentence

total_no_sentence = cal_num_sentence_in_emails(1,40)

def split_dataset(first_split,second_split):
    #Split data set into validation and training set depending on first_split and second_split
    #first split is the number of sentence before the first split
    #second split is ... before the second split
    #second_split = 0 if taking the first 10% or last 10% of training set to be validation set 

    if second_split == 0:
        validation_set_vector, training_set_vector = aligned_sentence_vectors[0:first_split], aligned_sentence_vectors[first_split:total_no_sentence]
        validation_set_score, training_set_score = aligned_scores[0:first_split], aligned_scores[first_split:total_no_sentence]
    
    else:
        validation_set_vector, training_set_vector = aligned_sentence_vectors[first_split:second_split],aligned_sentence_vectors[0:first_split] + aligned_sentence_vectors[second_split:total_no_sentence]
        validation_set_score, training_set_score = aligned_scores[first_split:second_split], aligned_scores[0:first_split] + aligned_scores[second_split:total_no_sentence]

    return validation_set_vector, validation_set_score, training_set_vector, training_set_score

def validation(split_position): 
    #perform validation based on the split_position
    #split_position will take the value of 4,8,12,16,20, ... 40

    first_split = 0 #first split is the number of sentence before the first split
    second_split = 0 #second split is ... before the second split 

    if split_position == 4:
        first_split = cal_num_sentence_in_emails(1,4)
        validation_set_vector, validation_set_score, training_set_vector, training_set_score = split_dataset(first_split,second_split)

    elif split_position == 40:
        first_split = cal_num_sentence_in_emails(37,40) 
        validation_set_vector, validation_set_score, training_set_vector, training_set_score = split_dataset(first_split,second_split)
        print "test"
        return first_split

    else: 
        first_split = cal_num_sentence_in_emails(1,split_position-4)
        second_split = cal_num_sentence_in_emails(1,split_position)
        validation_set_vector, validation_set_score, training_set_vector, training_set_score = split_dataset(first_split,second_split)

    random_forest = vectorize_bc3.server_train_classifier(training_set_vector,training_set_score)

    validation_set_predicted_score = [] 

    for sentence in validation_set_vector:
        validation_set_predicted_score.append(random_forest.predict(sentence))

    
    



