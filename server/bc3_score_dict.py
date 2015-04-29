#bc3_score_dict.py

import xml.etree.ElementTree as ET
import copy
from data.vectorized_bc3_data import *

from handy_function import *

tree  = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root  = tree.getroot() 

def bc3_score_dict():
	#Copy the structure of length_each_sentence
	bc3_score_dict = copy.deepcopy(num_word_in_each_sentence)

	current_sentence = 0 

	for thread in root:
		thread_listno = thread_listno_of_current_thread(thread)
		for sentence_ID in sorted(bc3_score_dict[thread_listno].keys()):
			bc3_score_dict[thread_listno][sentence_ID] = aligned_scores[current_sentence]
			current_sentence += 1	

	return bc3_score_dict

if __name__ == '__main__':
	#Load necessary pickled data
	f = open("data/num_word_in_each_sentence.pck", "r")
	num_word_in_each_sentence = pickle.load(f)
	f.close()
	#finish loading

	bc3_score_dict = bc3_score_dict()

	pickle_data(bc3_score_dict, "bc3_score_dict")