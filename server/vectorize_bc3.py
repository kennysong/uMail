import numpy as np
import xml.etree.ElementTree as ET
import string
import math
import collections
import copy
import pickle
import os
import re

from scipy.spatial import distance
from sklearn import ensemble
from sklearn.externals import joblib
from nltk.stem.snowball import SnowballStemmer
from helpers import *

def get_threads_in_root(root):
    '''Takes the root of an ElementTree return a list of threads (ET.Element).'''
    return list(root)

def get_emails_in_thread(thread):
    '''Takes a thread (ET.Element) return a list of emails (ET.Element).'''
    emails = []
    for i in range(len(thread)):
        if thread[i].tag == 'DOC':
            emails.append(thread[i])
    return emails

def get_email_of_sentence(sentence, emails):
    '''Given a list of emails (ET.Element), return the email (ET.Element) that
       the sentence (ET.Element) is in.'''
    for email in emails:
        sentences = get_sentences_in_email(email)
        if sentence in sentences:
            return email

def get_sentences_in_thread(thread):
    '''Given a thread (ET.Element), return a list of all sentences (ET.Element)
       in the thread.'''
    emails = get_emails_in_thread(thread)
    thread_sentences = []
    for email in emails:
        sentences = get_sentences_in_email(email)
        for sentence in sentences:
            thread_sentences.append(sentence)

    return thread_sentences

def get_sentences_in_email(email):
    '''If email is an ET.Element, returns a list of sentence objects.
       If email is a string, return a list of sentence strings.'''
    if isinstance(email, ET.Element):
        # If email is an ET.Element, return list of sentence objects
        text_tag = None
        for i in range(len(email)):
            if email[i].tag == 'Text':
                text_tag = email[i]
        return list(text_tag)
    else:
        # If email is a string, return list of sentence strings with punctuation at the end
        sentences_and_punctuation = re.split('([^a-zA-Z0-9][.?!+]|[.?!+][^a-zA-Z0-9]|\n+)', email)
        sentences = []
        for i in range(len(sentences_and_punctuation)):
            if re.match('([^a-zA-Z0-9][.?!+]|[.?!+][^a-zA-Z0-9]|\n+)', sentences_and_punctuation[i]) and len(sentences) != 0:
                # Append all trailing punctuation to previous sentence
                sentences[len(sentences) - 1] += sentences_and_punctuation[i]
            else:
                # Append a new sentence to the list
                sentences.append(sentences_and_punctuation[i])

        # Strip whitespace from beginning and end of sentence
        sentences = [sent.strip(' ') for sent in sentences]

        return sentences

def get_words_in_thread(thread):
    '''Returns a list of all the cleaned words in a thread (ET.Element).'''
    words = []
    emails = get_emails_in_thread(thread)
    for email in emails:
        words = get_words_in_email(email)
    return words

def get_words_in_email(email):
    '''Given an email as an ET.Element OR string, return a list of cleaned words.'''
    word_list = []
    sentences = get_sentences_in_email(email)
    for sentence in sentences:
        word_list += get_words_in_sentence(sentence)
    return word_list

def get_words_in_sentence(sentence):
    '''Given a sentence as an ET.Element OR string, return a list of cleaned words.'''
    if isinstance(sentence, ET.Element):
        sentence = sentence.text

    return clean_up_words(sentence.split())

def clean_up_words(word_list):
    '''Given list of words, returns list of lowercase, alphanumeric-only, singular words.'''

    # Make lowercase
    clean_word_list = [word.lower() for word in word_list]

    # Remove non alphabet or numbers
    for i in range(len(clean_word_list)):
        clean_word_list[i] = ''.join([t for t in clean_word_list[i] if (t in '1234567890' or t in string.ascii_lowercase)])

    # Make into stem words
    stemmer = SnowballStemmer("english")
    for i in range(len(clean_word_list)):
        clean_word_list[i] = stemmer.stem(clean_word_list[i])

    return clean_word_list

def get_num_recipients(email):
    '''Returns the number of recipients in an email (ET.Element).'''
    num_recipients = 0
    for email_tag in email:
        if email_tag.tag == 'To' or email_tag.tag == 'Cc':
            if email_tag.text is None: continue
            num_recipients += email_tag.text.count('@')
    return num_recipients

def get_rel_pos_in_email(sentence, email):
    '''Returns the # of sentences before sentence divided by total # in the email.
        - sentence can be an ET.Element and email can be a ET.Element.
        - sentence can be a string and an email can be a list of strings.'''

    # If sentence is an ET.Element, convert to string
    if isinstance(sentence, ET.Element):
        sentence = sentence.text

    # Get list of sentences as strings
    sentences = get_sentences_in_email(email)
    if isinstance(sentences[0], ET.Element):
        sentences = [s.text for s in sentences]

    # Return relative position of sentence
    if sentence in sentences:
        return sentences.index(sentence) / float(len(sentences)) * 100

def get_subject_similarity(subject, sentence):
    '''Returns the # words similarity of a sentence and the subject.
       The sentence and sentence can be a ET.Element or a string.'''

    # Change sentence to string
    if isinstance(sentence, ET.Element):
        sentence = sentence.text
    if isinstance(subject, ET.Element):
        subject = subject.text

    # Split into lists of words
    subject_words = get_words_in_sentence(subject)
    sentence_words = get_words_in_sentence(sentence)

    # Get # of words occurring in both
    overlap = len(set(subject_words).intersection(set(sentence_words)))
    return overlap

def get_tf_idf_vector(sentence, thread, root, word_counts_per_thread, vocab_list_index):
    '''Given a sentence and metadata, return a tf-idf vector of the sentence words.
        - idf = # of threads in entire corpus that have the word.
        - vocab_list_index = {word: index} for all the word in the corpus.
        - The tf-idf vectors have length(vocab_list_index) since the centroid is the avg vector.'''

    # Get cleaned words in the thread and the sentence
    threads = get_threads_in_root(root)
    thread_words = get_words_in_thread(thread)
    sentence_words = get_words_in_sentence(sentence)

    # Calculate term-frequency, number of times word appears in thread, normalized by len(thread)
    tf_vector = np.zeros(len(vocab_list_index))
    for word in sentence_words:
        occurrences = word_counts_per_thread[thread][word]
        tf = occurrences / float(len(thread_words))
        word_index = vocab_list_index[word]
        tf_vector[word_index] = tf

    # Calculate inverse document frequency, # of threads in entire corpus that have the word
    idf_vector = np.zeros(len(vocab_list_index))
    threads_words = [set(get_words_in_thread(thread)) for thread in threads]
    for word in sentence_words:
        num_threads_with_word = 0.0
        for thread_words in threads_words:
            if word in thread_words: num_threads_with_word += 1
        idf = math.log(len(threads) / num_threads_with_word)
        word_index = vocab_list_index[word]
        idf_vector[word_index] = idf

    # Get tf*idf vector
    tf_idf_vector = tf_vector * idf_vector

    return tf_idf_vector

def get_tf_local_idf_vector(sentence, thread, root, word_counts_per_thread, vocab_list_index):
    '''Given a sentence and metadata, return a tf-local-idf vector of the sentence words.
       idf = # of emails that have the word.
       vocab_list_index = {word: index} for all the word in the corpus.
       The tf-local-idf vectors have length(vocab_list_index) since the centroid is the avg vector.'''

    # Get cleaned words in the thread and the sentence
    thread_words = get_words_in_thread(thread)
    sentence_words = get_words_in_sentence(sentence)

    # Calculate term-frequency, number of times word appears in thread, normalized by len(thread)
    tf_vector = np.zeros(len(vocab_list_index))
    for word in sentence_words:
        occurrences = word_counts_per_thread[thread][word]
        tf = occurrences / float(len(thread_words))
        word_index = vocab_list_index[word]
        tf_vector[word_index] = tf

    # Calculate local inverse document frequency, # of emails in thread that have the word
    idf_vector = np.zeros(len(vocab_list_index))
    emails = get_emails_in_thread(thread)
    emails_words = [set(get_words_in_email(email)) for email in emails]
    for word in sentence_words:
        num_emails_with_word = 0.0
        for email_words in emails_words:
            if word in email_words: num_emails_with_word += 1
        idf = math.log(len(emails) / num_emails_with_word)
        word_index = vocab_list_index[word]
        idf_vector[word_index] = idf

    # Get tf*idf vector
    tf_idf_vector = tf_vector * idf_vector

    return tf_idf_vector

def get_annotation_from_thread(thread):
    annotations = []
    for i in range(len(thread)):
        if thread[i].tag == "annotation":
            annotations.append(thread[i])
    return annotations

def get_sentences_from_annotations(annotation):
    sentences = []
    for i in range(len(annotation)):
        if annotation[i].tag == "sentences":
            sentences.append(annotation[i])
    return sentences

def get_sentenceID_from_sentences(sentence):
    sentenceID = []
    for item in sentence:
        sentenceID.append(item.attrib["id"])
    return sentenceID

def get_sentenceID_from_thread(thread):
    sentenceID = []
    annotations = get_annotation_from_thread(thread)
    sentences = []

    for annotation in annotations:
        sentences.append(get_sentences_from_annotations(annotation))

    for sentence in sentences:
        sentenceID.append(get_sentenceID_from_sentences(sentence[0]))

    sentenceID = [sentenceID[i][j] for i in range(len(sentenceID)) for j in range(len(sentenceID[i]))]

    return sentenceID

def normalize_score(score):
    tree_corpus = ET.parse(os.path.dirname(os.path.abspath(__file__)) + "/bc3_corpus/bc3corpus.1.0/corpus.xml")
    root_corpus = tree_corpus.getroot()

    corpus_sentence = dict()
    for thread in root_corpus:
        sentence_id = dict()
        sentences = get_sentences_in_thread(thread)
        for sentence in sentences:
            sentence_id[sentence.attrib['id']] = sentence.text

        corpus_sentence[thread[1].text] = sentence_id

    threadID_in_corpus = score.keys()

    for thread in threadID_in_corpus:
        sentenceID_in_thread = score[thread].keys()
        for sentence in sentenceID_in_thread:
            score[thread][sentence] = float(score[thread][sentence])/len(corpus_sentence[thread][sentence].split() )

    return score

def get_annotated_scores():
    tree_annotation = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/bc3_corpus/bc3corpus.1.0/annotation.xml')
    root_annotation = tree_annotation.getroot()

    tree_corpus = ET.parse(os.path.dirname(os.path.abspath(__file__)) + "/bc3_corpus/bc3corpus.1.0/corpus.xml")
    root_corpus = tree_corpus.getroot()

    # Dictionary of thread number, each of which is another dictionary of the ID of each sentences in a thread
    corpus = dict()

    for thread in root_corpus:
        sentence_id = dict()
        sentences = get_sentences_in_thread(thread)
        for sentence in sentences:
            sentence_id[sentence.attrib['id']] = 0

        corpus[thread[1].text] = sentence_id

    score = copy.deepcopy(corpus)

    for thread in root_annotation:
        # Getting all the sentences in the 3 summaries in annotation file
        sentenceID = get_sentenceID_from_thread(thread)
        sentenceID_in_thread = score[thread[0].text].keys()
        for sentence in sentenceID_in_thread:
            score[thread[0].text][sentence] = sentenceID.count(sentence)

    score = normalize_score(score)

    return score

def vectorize_bc3_corpus():
    '''Vectorizes the entire BC3 corpus, returns a list of sentence feature vectors
       and a dictionary of {vector: vector/sentence metadata}.'''

    # Set up XML tree
    tree = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/bc3_corpus/bc3corpus.1.0/corpus.xml')
    root = tree.getroot()

    # Get word occurrence counts for each thread
    # This is used when calculating tf in tf-idf
    word_counts_per_thread = dict()
    for thread in root:
        words = get_words_in_thread(thread)
        word_counts = collections.Counter(words)
        word_counts_per_thread[thread] = word_counts

    print("Finished getting word counts for each thread")

    # Load bc3 vocab list variables from helper_variables.py
    from data.helper_variables import bc3_vocab_list as vocab_list
    from data.helper_variables import bc3_vocab_list_index as vocab_list_index

    print("Finished constructing vocab list")

    # Load tf_idf vector from pickled cache {thread_id-sentence_id: tf_idf vector}
    tf_idf_file = open(os.path.dirname(os.path.abspath(__file__)) + "/data/bc3_tf_idf_vectors", "rb")
    tf_local_idf_file = open(os.path.dirname(os.path.abspath(__file__)) + "/data/bc3_tf_local_idf_vectors", "rb")
    cached_tf_idf_vectors = pickle.load(tf_idf_file)
    cached_tf_local_idf_vectors = pickle.load(tf_local_idf_file)
    tf_idf_file.close()
    tf_local_idf_file.close()

    print("Loaded tf_idf vectors from pickle.")

    # Calculate and cache all centroid vectors, in {thread: centroid_vector}
    cached_centroid_vectors, cached_local_centroid_vectors = dict(), dict()

    for thread in root:
        tf_idf_vectors, tf_local_idf_vectors = [], []
        sentences = get_sentences_in_thread(thread)
        for sentence in sentences:
            # Get the key for the sentence as thread_id-sentence_id
            sentence_key = thread[1].text + '-' + sentence.attrib['id']

            tf_idf_vectors.append(cached_tf_idf_vectors[sentence_key])
            tf_local_idf_vectors.append(cached_tf_local_idf_vectors[sentence_key])

        centroid_vector = sum(tf_idf_vectors) / len(tf_idf_vectors)
        local_centroid_vector = sum(tf_local_idf_vectors) / len(tf_local_idf_vectors)

        cached_centroid_vectors[thread] = centroid_vector
        cached_local_centroid_vectors[thread] = local_centroid_vector

    print("Finished calculating centroid vectors")

    # Traverse the XML tree, vectorizing each sentence as we encounter it
    sentence_vectors = [] # [np.array, ...]
    sentence_vectors_metadata = [] # [(np.array, {metadata}), ...]
    for thread in root:
        # Information about this thread
        subject = thread[0]
        thread_id = thread[1].text
        emails = get_emails_in_thread(thread)
        thread_sentences = get_sentences_in_thread(thread)

        # Traverse each sentence in a thread, top to bottom
        for i in range(len(thread_sentences)):
            # Get some sentence metadata
            sentence = thread_sentences[i]
            current_email = get_email_of_sentence(sentence, emails)
            sentence_key = thread[1].text + '-' + sentence.attrib['id']
            num_recipients = get_num_recipients(current_email)
            email_number = emails.index(current_email) + 1
            tf_idf_vector = cached_tf_idf_vectors[sentence_key]
            tf_local_idf_vector = cached_tf_local_idf_vectors[sentence_key]
            centroid_vector = cached_centroid_vectors[thread]
            local_centroid_vector = cached_local_centroid_vectors[thread]

            # Vectorize the sentence and store it
            sentence_vector = vectorize_sentence(sentence, i, current_email, subject,
                              num_recipients, thread_sentences, email_number,
                              tf_idf_vector, tf_local_idf_vector,
                              centroid_vector, local_centroid_vector)
            sentence_vectors.append(sentence_vector)
            sentence_vectors_metadata.append((sentence_vector, {'id': sentence.attrib['id'], 'listno': thread_id}))

    return sentence_vectors, sentence_vectors_metadata

def vectorize_sentence(sentence, index, email, subject, num_recipients,
                       context_sentences, email_number, tf_idf_vector,
                       tf_local_idf_vector, centroid_vector, local_centroid_vector):
    '''Given a sentence and its metadata, return its feature vector.
       Arguments:
            - sentence is either a ET.Element or a string of a sentence.
            - index is the number of the sentence in the email (zero-indexed).
            - email is the email the sentence is in.
            - subject is the subject of the email, as a string or ET.Element.
            - num_recipients is the number of recipients.
            - context_sentences is all the sentences in either the thread or the email.
            - email_number is the number of the email within the thread.
            - tf_idf_vector is the vector of tf-idf values for all the sentence words.
            - tf_local_idf_vector is the vector of tf-local-idf values for all the sentence words.
            - centroid_vector is the average tf-idf vector in the thread.
            - local_centroid_vector is the average tf-local-idf vector in the thread.'''

    # Convert sentence to string if necessary
    if isinstance(sentence, ET.Element):
        sentence = sentence.text

    # Calculate all the (R14) features
    thread_line_number = index + 1
    rel_position_in_thread = index / float(len(context_sentences)) * 100
    length = len(get_words_in_sentence(sentence))
    is_question = 1 if '?' in sentence else 0
    rel_position_in_email = get_rel_pos_in_email(sentence, email)
    subject_similarity = get_subject_similarity(subject, sentence)

    tf_idf_sum = sum(tf_idf_vector)
    tf_idf_avg = sum(tf_idf_vector) / len(tf_idf_vector)

    centroid_similarity = 1 - distance.cosine(tf_idf_vector, centroid_vector)
    local_centroid_similarity = 1 - distance.cosine(tf_local_idf_vector, local_centroid_vector)

    #Custom feature: detect if there is a date or time or both in sentence
    date_time = detect_date_time(sentence)
     
    #Custom feature: detect if there is an email in sentence
    email_exist = detect_email(sentence)

    # Put all of these features into a vector
    sentence_vector = np.array([thread_line_number, rel_position_in_thread, centroid_similarity,
                      local_centroid_similarity, length, tf_idf_sum, tf_idf_avg, is_question,
                      email_number, rel_position_in_email, subject_similarity, num_recipients, date_time, date_time, date_time, date_time, date_time, date_time, date_time, date_time, email_exist])

    # Change NaN features to 0
    # This happens because one of the tf-idf vectors is all zero, because the
    # idf vector is all zero, i.e. the sentence words never appear in the corpus
    sentence_vector = [(0 if math.isnan(f) else f) for f in sentence_vector]

    return sentence_vector

def get_bc3_vectors_and_scores():
    '''Returns a list of vectors and scores that are aligned.'''
    sentence_vectors, sentence_vectors_metadata = vectorize_bc3_corpus()
    scores = get_annotated_scores()

    # Align these by index
    aligned_sentence_vectors = []
    aligned_scores = []
    for element in sentence_vectors_metadata:
        sentence_vector, metadata = element[0], element[1]
        thread_id = metadata['listno']
        sentence_id = metadata['id']
        score = scores[thread_id][sentence_id]

        aligned_sentence_vectors.append(sentence_vector)
        aligned_scores.append(score)

    return aligned_sentence_vectors, aligned_scores

def train_classifier(sent_vectors, sent_scores):
    '''Trains a classifier, returns RandomForest object.'''
    random_forest = ensemble.RandomForestRegressor()
    random_forest.fit(sent_vectors, sent_scores)

    return random_forest

if __name__ == '__main__':
    aligned_sentence_vectors, aligned_scores = get_bc3_vectors_and_scores()
