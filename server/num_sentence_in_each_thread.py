import xml.etree.ElementTree as ET

from handy_function import *

tree  = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root  = tree.getroot() 

def num_sentence_in_text(text):
	return len(text)

def num_sentence_in_doc(doc):
	for text in doc:
		if text.tag == "Text":
			return num_sentence_in_text(text)

def num_sentence_in_thread(thread):
	num_sentence = 0

	for doc in thread:
		if doc.tag == "DOC": num_sentence += num_sentence_in_doc(doc)

	return num_sentence

if __name__ == '__main__':
	num_sentence_in_each_thread = dict()

	for thread in root:
		for branch in thread:
			if branch.tag == "listno":
				thread_listno = branch.text
		num_sentence_in_each_thread[thread_listno] = num_sentence_in_thread(thread)

	pickle_data(num_sentence_in_each_thread, "num_sentence_in_each_thread")

