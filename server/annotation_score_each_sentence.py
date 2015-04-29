import xml.etree.ElementTree as ET
import operator

from handy_function import *

#global variable: annotation_score_each_sentence
    #annotation_score_each_sentence is a dictionary with the format 
    #[thread_listno] = {} (sub dictionary)
    #for sub dictionary
    #keys being the sentence ID and 
    #values being the no. of annotators that pick this sentence

tree = ET.parse("bc3_corpus/bc3corpus.1.0/annotation.xml")
root = tree.getroot() 

def return_sentence_ID(sentences):
    #return the sentence ID contained in the <sentences> tag in corpus as a list

    sentences_ID = [sent.attrib["id"] for sent in sentences]

    return sentences_ID

def update_annotation_score_each_sentence_with_sentenceID(annotation_score_each_sentence, thread_listno, sentences_ID):
    #update annotation_score_each_sentence(type:dict) with sentences_ID(type:list)
    #update means: add the sentence into the annotation_score_each_sentence if not present, if present annotation += 1

    for ID in sentences_ID:
        if ID not in annotation_score_each_sentence[thread_listno].keys():
            annotation_score_each_sentence[thread_listno][ID] = 1
        else:
            annotation_score_each_sentence[thread_listno][ID] += 1

def update_annotation_score_each_sentence_with_annotation(annotation_score_each_sentence, thread_listno, annotation):
    #update annotation_score_each_sentence(type:dict) with <annotation>(type:tree)
    #update means: add the sentence into the annotation_score_each_sentence if not present, if present annotation += 1

    for sentences in annotation:
        if sentences.tag == "sentences":
            sentences_ID =  return_sentence_ID(sentences)
    update_annotation_score_each_sentence_with_sentenceID(annotation_score_each_sentence,thread_listno,sentences_ID)

def update_annotation_score_each_sentence_with_each_thread(annotation_score_each_sentence,thread):
    #update annotation_score_each_sentence(type:dict) with <thread>(type:tree)

    #Determine the thread_listno of current thread
    for listno in thread:
        if listno.tag == "listno": thread_listno = listno.text

    #update annotation_score_each_sentence(type:dict) with each <annotation> in <thread>
    for annotation in thread:
        if annotation.tag == "annotation":
            update_annotation_score_each_sentence_with_annotation(annotation_score_each_sentence,thread_listno,annotation)

def update_annotation_score_each_sentence(annotation_score_each_sentence):
	#update annotation_score_each_sentence(type:dict) with <root>
    for thread in root:
        for listno in thread:
            if listno.tag == "listno":
                thread_listno = listno.text
                annotation_score_each_sentence[thread_listno] = dict()

        update_annotation_score_each_sentence_with_each_thread(annotation_score_each_sentence,thread)
    
if __name__ == '__main__':

    #Loading necessary data from pickled file
    f = open("data/num_word_in_each_sentence.pck", "r")
    num_word_in_each_sentence_data = pickle.load(f)
    #Finish loading data

    annotation_score_each_sentence = dict()

    #update annotation_score_each_sentence with annotation score for each sentence
    update_annotation_score_each_sentence(annotation_score_each_sentence)

    #normalization by sentence length (no. of words)
    for thread in annotation_score_each_sentence:
        for sentence in annotation_score_each_sentence[thread]:
            annotation_score_each_sentence[thread][sentence] = (annotation_score_each_sentence[thread][sentence])/(num_word_in_each_sentence_data[thread][sentence])

    pickle_data(annotation_score_each_sentence, "annotation_score_each_sentence")

