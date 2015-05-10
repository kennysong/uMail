import xml.etree.ElementTree as ET
import operator
import cPickle as pickle
import copy
import os
import nltk
import re
import datetime

import vectorize_bc3 as bc3

from dateutil import parser
from sklearn import ensemble

def listno_of_thread(thread):
    '''Get thread_listno of current thread.
       Debugged by Kenny on May 8th.'''
    for element in thread:
        if element.tag == "listno": return element.text

def pickle_data(data, file_name):
    '''Pickle data into /data as file_name.pck.
       Debugged by Kenny on May 8th.'''
    data_location = os.path.dirname(os.path.abspath(__file__)) + "/data/" + file_name + ".pck"
    f = open(data_location, "w")
    pickle.dump(data, f)
    f.close()

def pickle_load(file_name):
    '''Returns an unpickled object from file_name.pck.
       Debugged by Kenny on May 8th.'''
    data_location = os.path.dirname(os.path.abspath(__file__)) + "/data/" + file_name + ".pck"
    f = open(data_location, "r")
    loaded_data = pickle.load(f)
    f.close()
    return loaded_data

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a deep copy.
       Debugged by Kenny on May 8th.'''
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

def anno_scores_in_thread(thread):
    '''Return the annotation score of each sentence in a <thread> as a dictionary.
       Debugged by Kenny on May 8th.'''
    anno_scores = dict()
    for element in thread:
        if element.tag == "annotation":
            annotation_sent_IDs = sent_ID_in_annotation(element)
            for sent_ID in annotation_sent_IDs:
                if sent_ID in anno_scores: anno_scores[sent_ID] += 1
                else: anno_scores[sent_ID] = 1

    return anno_scores

def empty_bc3_structure_dict():
    '''Returns a dictionary that models the BC3 data structure:
       {thread_listno: {sentence_id: None, sentence_id2: None, ...}, ...}'''

    # Iterate through each thread
    bc3_structure_dict = dict()
    root_corpus = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/bc3_corpus/bc3corpus.1.0/corpus.xml').getroot()
    for thread in root_corpus:
        # Iterate through each sentence in the thread
        sentences = bc3.get_sentences_in_thread(thread)
        sentence_dict = {sentence.attrib['id']: None for sentence in sentences}
        thread_listno = listno_of_thread(thread)
        bc3_structure_dict[thread_listno] = sentence_dict

    return bc3_structure_dict

def return_num_word_each_sent(root_corpus):
    "Return number of word in each sentence in corpus in dictionary form"
    num_word_each_sent = dict()

    for thread in root_corpus:
        thread_listno = listno_of_thread(thread)
        num_word_each_sent[thread_listno] = num_word_each_sent_thread(thread)

    return num_word_each_sent

def return_anno_sent_scores(annotation_root, num_word_each_sent):
    '''Return the annotation score of each sentence in corpus (currently not normalized!).
       Debugged by Kenny on May 8th.'''

    # Get the annotated score for each sentence in the corpus
    anno_sent_scores = dict()
    for thread in annotation_root:
        thread_listno = listno_of_thread(thread)
        anno_sent_scores[thread_listno] = anno_scores_in_thread(thread)

    # Normalize these annotated scores
    # Note: currently disabling normalization of score to debug weighted recall!!
    # for thread_listno in anno_sent_scores:
    #     for sent_ID in anno_sent_scores[thread_listno]:
    #         anno_sent_scores[thread_listno][sent_ID] = anno_sent_scores[thread_listno][sent_ID] / num_word_each_sent[thread_listno][sent_ID]

    return anno_sent_scores

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
        thread_listno = listno_of_thread(thread)
        num_sent_each_thread[thread_listno] = num_sent_in_thread(thread)

    return num_sent_each_thread

def return_bc3_vector_dict(root_corpus, num_word_each_sent, sentence_vectors):
    "return the vectorized sentence in bc3 in dictionary form"

    bc3_vector_dict = empty_bc3_structure_dict()
    current_sentence = 0

    for thread in root_corpus:
        thread_listno = listno_of_thread(thread)
        for sent_ID in sorted(bc3_vector_dict[thread_listno].keys()):
            bc3_vector_dict[thread_listno][sent_ID] = sentence_vectors[current_sentence]
            current_sentence += 1

    return bc3_vector_dict

def return_bc3_score_dict(root_corpus, num_word_each_sent, scores):
    "Return the score of the vectorized sentence in bc3 in dictionary form"

    bc3_score_dict = empty_bc3_structure_dict()
    current_sentence = 0

    for thread in root_corpus:
        thread_listno = listno_of_thread(thread)
        for sent_ID in sorted(bc3_score_dict[thread_listno].keys()):
            bc3_score_dict[thread_listno][sent_ID] = scores[current_sentence]
            current_sentence += 1

    return bc3_score_dict

def sort_dict_by_value(dict):
    '''Sort a dictionary by values in descending order and turn into a tuple.'''
    dict_sorted = sorted(dict.items(), key=operator.itemgetter(1), reverse = True)

    return dict_sorted

def get_thread_summary(thread_listno, sent_scores, num_word_each_sent):
    '''Return summary of a thread based on sent_scores with the word limit, as a list of sentence IDs.
       This is for BC3 emails only, not for new emails!
       Debugged by Kenny on May 8th.'''

    # The word limit of a summary - 30% of original email by word count
    word_limit = sum(num_word_each_sent[thread_listno].values()) * 0.3

    # Sort each sentence in a <thread> according to annotation score in descending order
    # These are actually sentence IDs, not full sentences
    sent_sorted_by_importance = sort_dict_by_value(sent_scores[thread_listno])

    # Construct the summary with the word limit
    summary, summary_word_count = [], 0
    for sent in sent_sorted_by_importance:
        summary.append(sent[0])
        summary_word_count +=  num_word_each_sent[thread_listno][sent[0]]
        if summary_word_count >= word_limit: break

    return summary

def return_ideal_summaries(anno_sent_scores, num_word_each_sent):
    '''Return the ideal summary from the annotation score for each <thread> in dictionary form.
       Debugged by Kenny on May 8th.'''
    ideal_summaries = dict()
    for thread_listno in anno_sent_scores:
        ideal_summaries[thread_listno] = get_thread_summary(thread_listno, anno_sent_scores, num_word_each_sent)
    return ideal_summaries

def clear_helper_variables():
    '''Delete everything in the file data/helper_variables.py.
       Debugged by Kenny on May 8th.'''

    f = open(os.path.dirname(os.path.abspath(__file__)) + "/data/helper_variables.py", "w")
    f.write("")
    f.close()

def save_variable(variable, variable_name = str):
    '''Save the variable into data/helper_variables.
       Debugged by Kenny on May 8th.'''

    f = open(os.path.dirname(os.path.abspath(__file__)) + "/data/helper_variables.py", "a")
    f.write(variable_name + " = " + repr(variable) + "\n\n")
    f.close()

def save_cross_validation_variables():
    '''Saves all variables needed in cross_validation.py to helper_variables.py and pickles.'''
    tree = ET.parse(os.path.dirname(os.path.abspath(__file__)) + "/bc3_corpus/bc3corpus.1.0/annotation.xml")
    annotation_root = tree.getroot()

    tree = ET.parse(os.path.dirname(os.path.abspath(__file__)) + "/bc3_corpus/bc3corpus.1.0/corpus.xml")
    root_corpus = tree.getroot()

    # Vectorize the bc3 corpus
    from vectorize_bc3 import get_bc3_vectors_and_scores
    sentence_vectors, scores = get_bc3_vectors_and_scores()

    # num_word_each_sent is the number of words in each sentence
    num_word_each_sent = return_num_word_each_sent(root_corpus)
    save_variable(num_word_each_sent, "num_word_each_sent")

    # anno_sent_scores is the normalized annotation score for each sentence
    anno_sent_scores = return_anno_sent_scores(annotation_root, num_word_each_sent)
    save_variable(anno_sent_scores, "anno_sent_scores")

    # num_sent_each_thread is the number of sentence in each thread
    num_sent_each_thread = return_num_sent_each_thread(root_corpus)
    save_variable(num_sent_each_thread, "num_sent_each_thread")

    # bc3_vector_dict is the bc3_vectorized vectors in dictionary format
    bc3_vector_dict = return_bc3_vector_dict(root_corpus, num_word_each_sent, sentence_vectors)
    save_variable(bc3_vector_dict, "bc3_vector_dict")

    # bc3_score_dict is the bc_3vectorized scores in dictionary format
    bc3_score_dict = return_bc3_score_dict(root_corpus, num_word_each_sent, scores)
    save_variable(bc3_score_dict, "bc3_score_dict")

    # Train classifier on full BC3 corpus and return the trained classifier 
    random_forest_full = train_classifier(sentence_vectors, scores)
    pickle_data(random_forest_full, "random_forest_full")

    # Dictionary of ideal summaries weighted across 3 annotators
    ideal_summaries = return_ideal_summaries(anno_sent_scores, num_word_each_sent)
    save_variable(ideal_summaries, 'ideal_summaries')

def save_vectorize_bc3_variables():
    '''Saves all variables needed in vectorize_bc3.py to helper_variables.py
       and pickles.'''

    # Import some helper functions from vectorize_bc3.py
    from vectorize_bc3 import get_words_in_thread

    # Calculate vocab list for entire corpus
    # This is needed for the tf-idf vectors, see function for more details
    tree = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/bc3_corpus/bc3corpus.1.0/corpus.xml')
    root = tree.getroot()

    vocab_list, vocab_list_set = [], set()
    for thread in root:
        thread_vocab_list = get_words_in_thread(thread)
        for item in thread_vocab_list:
            if item not in vocab_list_set:
                vocab_list.append(item)
                vocab_list_set.add(item)
    vocab_list_index = {word: index for index, word in enumerate(vocab_list)}

    save_variable(vocab_list, 'bc3_vocab_list')
    save_variable(vocab_list_index, 'bc3_vocab_list_index')

    ## Uncomment this to recalculate the tf_idf vector pickle cache (takes 4-5 hours)
    # # Calculate and cache all tf_idf vectors, in {sentence: tf-idf vector}
    # cached_tf_idf_vectors, cached_tf_local_idf_vectors = dict(), dict()
    # for thread_index in range(len(root)):
    #         # Get tf-idf vector for each sentence in this thread
    #         thread = root[thread_index]
    #         thread_sentences = get_sentences_in_thread(thread)
    #         for sentence in thread_sentences:
    #                 # Get the key for the sentence as thread_id-sentence_id
    #                 thread_id = thread[1].text
    #                 sentence_id = sentence.attrib['id']
    #                 sentence_key = thread_id + '-' + sentence_id

    #                 tf_idf_vector = get_tf_idf_vector(sentence, thread, root, word_counts_per_thread, vocab_list_index)
    #                 cached_tf_idf_vectors[sentence_key] = tf_idf_vector
    #                 tf_local_idf_vector = get_tf_local_idf_vector(sentence, thread, root, word_counts_per_thread, vocab_list_index)
    #                 cached_tf_local_idf_vectors[sentence_key] = tf_local_idf_vector

    #         print("Calculated tf-idf for thread %i" % thread_index)

    # # Pickle the tf-idf dictionaries
    # tf_idf_file = open(os.path.dirname(os.path.abspath(__file__)) + "/data/bc3_tf_idf_vectors", "wb")
    # pickle.dump(cached_tf_idf_vectors, tf_idf_file, protocol=2)
    # tf_idf_file.close()
    # tf_local_idf_file = open(os.path.dirname(os.path.abspath(__file__)) + "/data/bc3_tf_local_idf_vectors", "wb")
    # pickle.dump(cached_tf_local_idf_vectors, tf_local_idf_file, protocol=2)
    # tf_local_idf_file.close()

    # print("Successfully pickled tf-idf")

def save_vectorize_email_variables():
    '''Saves all variables needed in vectorize_email.py to helper_variables.py'''

    # Import some helper functions from vectorize_bc3.py
    from vectorize_bc3 import get_threads_in_root, get_bc3_vectors_and_scores, get_words_in_thread

    # Calculate words in each thread, save to threads_words.py
    tree = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/bc3_corpus/bc3corpus.1.0/corpus.xml')
    root_corpus = tree.getroot()
    threads = get_threads_in_root(root_corpus)
    threads_words = [set(get_words_in_thread(thread)) for thread in threads]
    save_variable(threads_words, 'bc3_threads_words')

    # Vectorize the BC3 corpus, and write to vectorized_bc3_data.py
    sentence_vectors, scores = get_bc3_vectors_and_scores()
    save_variable(sentence_vectors, 'bc3_sentence_vectors')
    save_variable(scores, 'bc3_scores')

def save_all_variables():
    '''Recalculates and saves all variables needed in cross_validation.py, 
       vectorize_bc3.py, vectorize_email.py in helper_variables.py and pickles.
       Do not change the order!'''

    # Empty the helpers_variables.py file
    clear_helper_variables()

    # Start re-writing variables to helpers_variables.py (and pickles)
    print('In save_vectorize_bc3_variables()')
    save_vectorize_bc3_variables()
    print('In save_cross_validation_variables()')
    save_cross_validation_variables()
    print('In save_vectorize_email_variables()')
    save_vectorize_email_variables()
    print("Completed save_all_variables().")

def remove_adverb(string):
    "Take in a string, remove adverbs from the strings, return the strings without adverbs"

    #Tokenize string
    string_tokenized = nltk.word_tokenize(string)

    #Tag each tokens with its sentence functions
    string_tagged = nltk.pos_tag(string_tokenized)

    adverbs = []

    for word in string_tagged:
        if word[1] == "RB": adverbs.append(word[0])

    remove = '|'.join(adverbs)
    regex = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)
    string = regex.sub("", string)

    return string

def detect_date_time(string):
    "Return True if there is mention of days, date or time in a string"
    import datetime
    default =  datetime.datetime.utcnow()

    #A hacky way of pre-processing the string: If the string contains 30-35 (dd-dd with dd>30), the dateutil parse will return an error.
    for i in range(len(string)):
        try:
            if string[i] == "-":
                try: 
                    if int(string[i-2:i]) > 12 and int(string[i+1:i+3]) > 12:
                        string = string.replace(string[i-2:i+1],"")
                except ValueError: 
                    pass
        except IndexError: 
            return True

    #default is set to datetime.datetime.utcnow below because of the way the class parser works. 
    #It returns the default time (curente date 00:00:00) if there is no date and time in the string
    #This is bad if we have if the user chooses to summarize an email that contains the date
    #in which the users choose to do so
    try:
        datetime = parser.parse(string, yearfirst = True, default = default, fuzzy = True)
    except ValueError: 
        pass

    # print "datetime is " 
    # print datetime
    #need to have type error below because of the existence of sentence that include information about timezone in bc3 corpus
    try:
        if datetime == default: 
            return False
        else: 
            return True
    except TypeError: return True

def detect_email(string):
    "Return 1 if there is an email in the string. Return 0 otherwise"
    return 1 if "@" in string else 0

def get_num_you(sentence_words):
    '''Given a list of uncleaned sentence words, returns the number of "you"s in the sentence.'''
    num_you = 0
    for word in sentence_words:
        if "you" == word[:3]:
            num_you += 1
    return num_you

def get_num_i(sentence_words):
    '''Given a list of uncleaned sentence words, returns the number of "I"s in the sentences.'''
    num_i = 0
    for word in sentence_words:
        if ("i" == word) or ("i'" == word[:2]):
            num_i += 1
    return num_i

def train_classifier(sent_vectors, sent_scores):
    '''Trains a classifier, returns RandomForest object.'''
    random_forest = ensemble.RandomForestRegressor()
    random_forest.fit(sent_vectors, sent_scores)

    return random_forest

if __name__ == '__main__':
    save_all_variables()