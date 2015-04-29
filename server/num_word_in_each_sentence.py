import xml.etree.ElementTree as ET

from handy_function import *

tree = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root = tree.getroot()

#Global variable: 
	# num_word_in_each_sentence is a dictionary with the format
	# [thread_listno] = {} (sub dictionary)
	# the sub dictionary has the format
	# [sentence_ID] = sentence_num_word_in_each_sentence 

def update_num_word_in_each_sentence_by_each_text(num_word_in_each_sentence, thread_listno, text):
    #update num_word_in_each_sentence(type:dict) with the num_word_in_each_sentence of the sentence in <text> that belongs to the <thread> with thread_listno
 
    for sent in text: 
        sentence_length = float(len(sent.text.split()))
        num_word_in_each_sentence[thread_listno][sent.attrib["id"]] = sentence_length

def update_num_word_in_each_sentence_by_each_doc(num_word_in_each_sentence, thread_listno, doc):
    #update num_word_in_each_sentence(type:dict) with the num_word_in_each_sentence of the sentences in <doc> that belongs to the <thread> with thread_listno

    for text in doc:
        if text.tag == "Text":
            update_num_word_in_each_sentence_by_each_text(num_word_in_each_sentence,thread_listno,text)

def update_num_word_in_each_sentence_by_each_thread(num_word_in_each_sentence, thread):
    #update num_word_in_each_sentence(type:dict) with the num_word_in_each_sentence of sentences in a <thread>

    #determine the thread_listno of current thread
    for branch in thread:
        if branch.tag == "listno": thread_listno = branch.text

    #update num_word_in_each_sentence(type:dict) with the num_word_in_each_sentence of each <doc> in <thread>
    for doc in thread:
        if doc.tag == "DOC":
            update_num_word_in_each_sentence_by_each_doc(num_word_in_each_sentence,thread_listno,doc)

def num_word_in_each_sentence_in_root(num_word_in_each_sentence):
    #update num_word_in_each_sentence(type:dic) with the num_word_in_each_sentence of sentences in <root>
    
    for thread in root: 
        for branch in thread:
            if branch.tag == "listno": 
                thread_listno = branch.text
                num_word_in_each_sentence[thread_listno] = dict()
    
        update_num_word_in_each_sentence_by_each_thread(num_word_in_each_sentence,thread)

if __name__ == '__main__':
    num_word_in_each_sentence = dict()
    num_word_in_each_sentence_in_root(num_word_in_each_sentence)

    pickle_data(num_word_in_each_sentence, "num_word_in_each_sentence")