import xml.etree.ElementTree as ET
import pickle

tree = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root = tree.getroot()

#length is a global dictionary with the format
#[thread_listno] = {} (sub dictionary)
#the sub dictionary has the format
#[sentence_ID] = sentence_length 

def update_length_by_each_text(length,thread_listno,text):
    #update length(type:dict) with the length of the sentence in <text> that belongs to the thread with thread_listno
 
    for sent in text: 
        sentence_length = float(len(sent.text.split()))
        length[thread_listno][sent.attrib["id"]] = sentence_length

def update_length_by_each_doc(length,thread_listno,doc):
    #update length(type:dic) with the length of the sentences in <doc> that belongs to the thread with thread_listno

    for text in doc:
        if text.tag == "Text":
            update_length_by_each_text(length,thread_listno,text)

def update_length_by_each_thread(length,thread):
    #update length(type:dic) with the length of sentences in a thread

    for branch in thread:
        if branch.tag == "listno": thread_listno = branch.text

    for doc in thread:
        if doc.tag == "DOC":
            update_length_by_each_doc(length,thread_listno,doc)


def return_length_each_sentence_in_root(length):
    #update length(type:dic) with the length of sentences in root
    
    for thread in root: 
        for branch in thread:
            if branch.tag == "listno": 
                thread_listno = branch.text
                length[thread_listno] = dict()
    
        update_length_by_each_thread(length,thread)

if __name__ == '__main__':
    length = dict()
    return_length_each_sentence_in_root(length)

    #pick length dictionary
    f = open("data/length_each_sentence.pck","w")
    pickle.dump(length,f)
    f.close()