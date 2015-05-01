import xml.etree.ElementTree as ET
import operator
import cPickle as pickle
import copy
from data.vectorized_bc3_data import *

def thread_listno_of_current_thread(thread):
    '''Get thread_listno of current thread.'''
    for listno in thread:
        if listno.tag == "listno":
            return listno.text

def pickle_data(data, file_name):
    '''Pickle data into /data as file_name.pck.'''
    data_location = "data/" + file_name + ".pck"
    f = open(data_location, "w")
    pickle.dump(data,f)
    f.close()

def pickle_load(file_name):
    '''Returns an unpickled object from file_name.pck.'''
    data_location = "data/" + file_name + ".pck"
    f = open(data_location, "r")
    return pickle.load(f)

def write_bc3_vocab_data(vocab_list, vocab_list_index):
    '''Writes vocab_list and vocab_list_index to bc3_vocab_data.py.'''
    f = open('data/bc3_vocab_data.py', 'w')
    f.write('# Words in BC3 corpus\n\n')
    f.write('vocab_list = ' + str(vocab_list) + '\n')
    f.write('vocab_list_index = ' + str(vocab_list_index) + '\n')
    f.close()

def write_vectorized_bc3_data(sentence_vectors, scores):
    '''Writes sentence_vectors and scores to vectorized_bc3_data.py.'''
    f = open('data/vectorized_bc3_data.py', 'w')
    f.write('# BC3 corpus as sentence vectors with their respective scores\n\n')
    f.write('scores = ' + str(scores) + '\n')
    f.write('sentence_vectors = ' + str(sentence_vectors) + '\n')
    f.close()

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a deep copy.'''
    z = copy.deepcopy(x)
    z.update(y)
    return z

def num_word_sent(sent):
    '''Return number of word of each sentence in dictionary form.'''
    num_word = dict()
    num_word[sent.attrib["id"]] = float(len(sent.text.split()))
    return num_word

def num_word_each_sent_doc(doc):
    '''Return number of word of each sentence in <DOC> in dictionary form.'''
    num_doc = dict()
    for text in doc:
        if text.tag == "Text":
            for sent in text:
                num_doc = merge_two_dicts(num_doc, num_word_sent(sent))

    return num_doc

def num_word_each_sent_thread(thread):
    '''Return number of word of each sentence in <thread> in dictionary form.'''
    num_thread = dict()
    for doc in thread:
        if doc.tag == "DOC":
            num_thread = merge_two_dicts(num_thread, num_word_each_sent_doc(doc))

    return num_thread

def return_sentence_ID(sentences):
    '''Return the sentence ID contained in the <sentences> tag in corpus as a list.'''
    sentences_ID = [sent.attrib["id"] for sent in sentences]
    return sentences_ID

def sent_ID_in_sentences(sentences):
    '''Return the sentence ID contained in the <sentences> tag in corpus_annotation as a list.'''
    sentences_ID = [sent.attrib["id"] for sent in sentences]
    return sentences_ID

def sent_ID_in_annotation(annotation):
    '''Return the sentence ID contained in the <annotation> tag in corpus_annotation as a list.'''
    annotation_ID = []
    for sentences in annotation:
        if sentences.tag == "sentences":
            annotation_ID = sent_ID_in_sentences(sentences)

    return annotation_ID

def anno_sent_ID_in_thread(thread):
    '''Return the annotation score of each sentence in a <thread> as a dictionary.'''
    anno_sent_in_thread = dict()
    for annotation in thread:
        if annotation.tag == "annotation":
            annotation_sent_ID = sent_ID_in_annotation(annotation)
            for sent_ID in annotation_sent_ID:
                if sent_ID in anno_sent_in_thread.keys():
                    anno_sent_in_thread[sent_ID] += 1
                else:
                    anno_sent_in_thread[sent_ID] = 1

    return anno_sent_in_thread

def return_num_word_each_sent(root_corpus):
    "Return number of word in each sentence in corpus in dictionary form"
    num_word_each_sent = dict()

    for thread in root_corpus:
        thread_listno = thread_listno_of_current_thread(thread)
        num_word_each_sent[thread_listno] = num_word_each_sent_thread(thread)

    return num_word_each_sent

def return_anno_score_sent(root_annotation, num_word_each_sent):
    "Return the annotation score of each sentence in corpus (normalized)"
    anno_score_sent = dict()

    for thread in root_annotation:
        thread_listno = thread_listno_of_current_thread(thread)
        anno_score_sent[thread_listno] = anno_sent_ID_in_thread(thread)

    for thread_listno in anno_score_sent:
        for sent_ID in anno_score_sent[thread_listno]:
            anno_score_sent[thread_listno][sent_ID] = anno_score_sent[thread_listno][sent_ID] / num_word_each_sent[thread_listno][sent_ID]

    return anno_score_sent

def num_sent_in_text(text):
    "return the number of sentence in a <text>"
    return len(text)

def num_sent_in_doc(doc):
    "return the number of sentence in a <DOC>"
    for text in doc:
        if text.tag == "Text":
            return num_sent_in_text(text)

def num_sent_in_thread(thread):
    "Return the number of sentence in a <thread>"
    num_sent = 0
    for doc in thread:
        if doc.tag == "DOC": num_sent += num_sent_in_doc(doc)

    return num_sent

def return_num_sent_each_thread(root_corpus):
    "return the number of sentence in corpus in dictionary form"
    num_sent_each_thread = dict()

    for thread in root_corpus:
        thread_listno = thread_listno_of_current_thread(thread)
        num_sent_each_thread[thread_listno] = num_sent_in_thread(thread)

    return num_sent_each_thread

def return_bc3_vector_dict(root_corpus, num_word_each_sent, sentence_vectors):
    "return the vectorized sentence in bc3 in dictionary form"

    # Copy the structure of num_word_each_sent
    bc3_vector_dict = copy.deepcopy(num_word_each_sent)

    current_sentence = 0

    for thread in root_corpus:
        thread_listno = thread_listno_of_current_thread(thread)
        for sent_ID in sorted(bc3_vector_dict[thread_listno].keys()):
            bc3_vector_dict[thread_listno][sent_ID] = sentence_vectors[current_sentence]
            current_sentence += 1

    return bc3_vector_dict

def return_bc3_score_dict(root_corpus, num_word_each_sent, scores):
    "Return the score of the vectorized sentence in bc3 in dictionary form"

    # Qns to clarify: where did we get these scores from ? 
    # Copy the structure of num_word_each_sent
    bc3_score_dict = copy.deepcopy(num_word_each_sent)

    current_sentence = 0

    for thread in root_corpus:
        thread_listno = thread_listno_of_current_thread(thread)
        for sent_ID in sorted(bc3_score_dict[thread_listno].keys()):
            bc3_score_dict[thread_listno][sent_ID] = scores[current_sentence]
            current_sentence += 1

    return bc3_score_dict

def sort_dict_by_value(dict):
    '''Sort a dictionary by values in descending order and turn into a tuple.'''
    dict_sorted = sorted(dict.items(), key=operator.itemgetter(1), reverse = True)
    return dict_sorted

def ideal_summary_thread(thread_listno, anno_score_sent, num_word_each_sent):
    '''Return ideal summary of a thread with the length limit.'''

    # length_limit is the length limit of ideal summary - 30% of original email by word count
    length_limit = sum(num_word_each_sent[thread_listno].values())/100*30

    # Sort each sentence in a <thread> according to annotation score
    sent_sorted_by_importance = sort_dict_by_value(anno_score_sent[thread_listno])

    ideal_summary_word_count = 0
    ideal_summary = []

    for sent in sent_sorted_by_importance:
        ideal_summary.append(sent[0])
        ideal_summary_word_count += ideal_summary_word_count + num_word_each_sent[thread_listno][sent[0]]
        if ideal_summary_word_count >= length_limit:
            break

    ideal_summary = sorted(ideal_summary)
    return ideal_summary

def return_ideal_summaries(anno_score_sent, num_word_each_sent):
    "Return the ideal summary from the annotaion score for each <thread> in dictionary form "
    ideal_summaries = dict()
    for thread_listno in anno_score_sent:
        ideal_summaries[thread_listno] = ideal_summary_thread(thread_listno, anno_score_sent, num_word_each_sent)
    return ideal_summaries

# if __name__ == '__main__':

tree = ET.parse("bc3_corpus/bc3corpus.1.0/annotation.xml")
root_annotation = tree.getroot()

tree = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
root_corpus = tree.getroot()

# num_word_each_sent is the number of words in each sentence
num_word_each_sent = return_num_word_each_sent(root_corpus)

# anno_score_sent is the normalized annotation score for each sentence
anno_score_sent = return_anno_score_sent(root_annotation, num_word_each_sent)

# num_sent_each_thread is the number of sentence in each thread
num_sent_each_thread = return_num_sent_each_thread(root_corpus)

# bc3_vector_dict is the bc3_vectorized vectors in dictionary format
bc3_vector_dict = return_bc3_vector_dict(root_corpus, num_word_each_sent, sentence_vectors)

# bc3_score_dict is the bc_3vectorized scores in dictionary format
bc3_score_dict = return_bc3_score_dict(root_corpus, num_word_each_sent, scores)

# Dictionary of ideal summaries weighted across 3 annotators
ideal_summaries = return_ideal_summaries(anno_score_sent, num_word_each_sent)