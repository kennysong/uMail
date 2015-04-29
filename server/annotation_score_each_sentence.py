import xml.etree.ElementTree as ET
import operator
import pickle
from data.length_each_sentence import *

tree = ET.parse("bc3_corpus/bc3corpus.1.0/annotation.xml")
root = tree.getroot() 

def return_sentence_ID(sentences):
    #return the sentence ID contained in the <sentences> in corpus as a list

    sentences_ID = [sent.attrib["id"] for sent in sentences]

    return sentences_ID

def update_summary_with_sentenceID(summary, thread_listno, sentences_ID):
    #update summary(type:dicti) with sentences_ID(type:list)
    #update means: add the sentence into the summary if not present, if present score += 1

    for ID in sentences_ID:
        if ID not in summary[thread_listno].keys():
            summary[thread_listno][ID] = 1
        else:
            summary[thread_listno][ID] += 1

def update_summary_with_annotation(summary, thread_listno, annotation):
    #update summary(type:dict) with <annotation>(type:tree)
    #update means: add the sentence into the summary if not present, if present score += 1

    for sentences in annotation:
        if sentences.tag == "sentences":
            sentences_ID =  return_sentence_ID(sentences)
    update_summary_with_sentenceID(summary,thread_listno,sentences_ID)

def update_summary_with_each_thread(summary,thread):
    #update summary(type:dic) with <thread>(type:tree)

    for listno in thread:
        if listno.tag == "listno": thread_listno = listno.text

    three_annotation = [branch for branch in thread if branch.tag == "annotation"] 
    for annotation in thread:
        if annotation.tag == "annotation":
            update_summary_with_annotation(summary,thread_listno,annotation)

def update_summary(summary):
    for thread in root:
        for listno in thread:
            if listno.tag == "listno":
                thread_listno = listno.text
                summary[thread_listno] = dict()

        update_summary_with_each_thread(summary,thread)
    
if __name__ == '__main__':
    #summary is a dictionary with the format 
    #[thread_listno] = {} (sub dictionary)
    #for sub dictionary
    #keys being the sentence ID and 
    #values being the no. of annotators that pick this sentence
    summary = dict()

    update_summary(summary)

    #normalization by sentence length

    for thread in summary:
        for sentence in summary[thread]:
            summary[thread][sentence] = (summary[thread][sentence])/(length_each_sentence[thread][sentence])

    print summary
