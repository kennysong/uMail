import xml.etree.ElementTree as ET
import operator
import cPickle as pickle
import copy
import nltk
import re

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

def get_thread_summary(thread_listno, sentence_scores, num_word_each_sent):
    '''Return summary of a thread with the length limit, as a list of sentence IDs.'''

    # length_limit is the length limit of ideal summary - 30% of original email by word count
    length_limit = sum(num_word_each_sent[thread_listno].values())/100*30

    # Sort each sentence in a <thread> according to annotation score
    sent_sorted_by_importance = sort_dict_by_value(sentence_scores[thread_listno])

    summary_word_count = 0
    summary = []

    for sent in sent_sorted_by_importance:
        summary.append(sent[0])
        summary_word_count += summary_word_count + num_word_each_sent[thread_listno][sent[0]]
        if summary_word_count >= length_limit:
            break

    summary = sorted(summary)
    return summary

def return_ideal_summaries(anno_score_sent, num_word_each_sent):
    '''Return the ideal summary from the annotaion score for each <thread> in dictionary form.'''
    ideal_summaries = dict()
    for thread_listno in anno_score_sent:
        ideal_summaries[thread_listno] = get_thread_summary(thread_listno, anno_score_sent, num_word_each_sent)
    return ideal_summaries

def clear_helper_variables():
    "delete everything in the file data/helper_variables.py"

    f = open("data/helper_variables.py", "w")
    f.close()

def save_variable(variable, variable_name = str):
    "Save the variable into data/helper_variables"

    f = open("data/helper_variables.py", "a")

    f.write(variable_name + " = " + repr(variable) + "\n" +"\n")

    f.close()

def save_cross_validation_variables():
    '''Saves all variables needed in cross_validation.py to helper_variables.py 
       and pickles.'''
    tree = ET.parse("bc3_corpus/bc3corpus.1.0/annotation.xml")
    root_annotation = tree.getroot()

    tree = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
    root_corpus = tree.getroot()

    # Vectorize the bc3 corpus
    from vectorize_bc3 import get_bc3_vectors_and_scores
    sentence_vectors, scores = get_bc3_vectors_and_scores()

    # num_word_each_sent is the number of words in each sentence
    num_word_each_sent = return_num_word_each_sent(root_corpus)
    save_variable(num_word_each_sent, "num_word_each_sent")

    # anno_score_sent is the normalized annotation score for each sentence
    anno_score_sent = return_anno_score_sent(root_annotation, num_word_each_sent)
    save_variable(anno_score_sent, "anno_score_sent")

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
    from vectorize_bc3 import train_classifier
    random_forest_full = train_classifier(sentence_vectors, scores)
    pickle_data(random_forest_full, "random_forest_full")

    # Dictionary of ideal summaries weighted across 3 annotators
    ideal_summaries = return_ideal_summaries(anno_score_sent, num_word_each_sent)
    save_variable(ideal_summaries, 'ideal_summaries')

def save_vectorize_bc3_variables():
    '''Saves all variables needed in vectorize_bc3.py to helper_variables.py
       and pickles.'''

    # Import some helper functions from vectorize_bc3.py
    from vectorize_bc3 import clean_up_words, get_words_in_thread

    # Calculate vocab list for entire corpus
    # This is needed for the tf-idf vectors, see function for more details
    tree = ET.parse('bc3_corpus/bc3corpus.1.0/corpus.xml')
    root = tree.getroot()

    vocab_list, vocab_list_set = [], set()
    for thread in root:
        thread_vocab_list = clean_up_words(get_words_in_thread(thread))
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
    # tf_idf_file = open("data/bc3_tf_idf_vectors", "wb")
    # pickle.dump(cached_tf_idf_vectors, tf_idf_file, protocol=2)
    # tf_idf_file.close()
    # tf_local_idf_file = open("data/bc3_tf_local_idf_vectors", "wb")
    # pickle.dump(cached_tf_local_idf_vectors, tf_local_idf_file, protocol=2)
    # tf_local_idf_file.close()

    # print("Successfully pickled tf-idf")

def save_vectorize_email_variables():
    '''Saves all variables needed in vectorize_email.py to helper_variables.py'''

    # Import some helper functions from vectorize_bc3.py
    from vectorize_bc3 import get_threads_in_root, clean_up_words, get_bc3_vectors_and_scores, get_words_in_thread

    # Calculate words in each thread, save to threads_words.py
    tree = ET.parse('bc3_corpus/bc3corpus.1.0/corpus.xml')
    root_corpus = tree.getroot()
    threads = get_threads_in_root(root_corpus)
    threads_words = [set(clean_up_words(get_words_in_thread(thread))) for thread in threads]
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
