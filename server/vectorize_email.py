import numpy as np
import xml.etree.ElementTree as ET
import string
import math
import re

import vectorize_bc3 as bc3

from scipy.spatial import distance
from helpers import *

def get_tf_idf_vector(sentence, email, email_words, sentence_words, threads_words, bc3_threads, vocab_list_index):
    '''Takes in a sentence and metadata, and returns a tf-idf vector of the sentence words.
       idf = # of threads in BC3 corpus that have the word.'''

    # Calculate term-frequency, number of times word appears in the email, normalized by len(email)
    tf_vector = np.zeros(len(vocab_list_index))
    for word in sentence_words:
        occurrences = len([email_word for email_word in email_words if word == email_word])
        tf = occurrences / float(len(email_words))
        word_index = vocab_list_index[word]
        tf_vector[word_index] = tf

    # Calculate inverse document frequency, # of threads in BC3 corpus that have the word
    # num_words_with_thread starts at 1.0, since it occurs in its own email
    # i.e. we can consider that email part of the corpus
    idf_vector = np.zeros(len(vocab_list_index))
    for word in sentence_words:
        num_threads_with_word = 1.0
        for thread_words in threads_words:
            if word in thread_words: num_threads_with_word += 1
        idf = math.log(len(bc3_threads) / num_threads_with_word)
        word_index = vocab_list_index[word]
        idf_vector[word_index] = idf

    # Get tf*idf vector
    tf_idf_vector = tf_vector * idf_vector

    return tf_idf_vector

def get_tf_local_idf_vector(sentence, email, email_words, sentence_words, vocab_list_index):
    '''Takes in a sentence and metadata, and returns a tf-local-idf vector of the sentence words.
       local-idf = # of emails that the word appears in = 1.'''

    # Calculate term-frequency, number of times word appears in the email, normalized by len(email)
    tf_vector = np.zeros(len(vocab_list_index))
    for word in sentence_words:
        occurrences = len([email_word for email_word in email_words if word == email_word])
        tf = occurrences / float(len(email_words))
        word_index = vocab_list_index[word]
        tf_vector[word_index] = tf

    # Calculate local inverse document frequency, # of emails in thread that have the word
    # This is always 1, since there is only one email in the "thread"
    idf_vector = np.zeros(len(vocab_list_index))
    for word in sentence_words:
        word_index = vocab_list_index[word]
        idf_vector[word_index] = 1

    # Get tf*idf vector
    tf_idf_vector = tf_vector * idf_vector

    return tf_idf_vector

def vectorize_email(email_message, title, to_cc):
    '''Takes in an email and metadata, returns list of feature vectors for sentences.'''

    # Split email into list of sentences
    sentences = bc3.get_sentences_in_email(email_message)

    # Load the BC3 corpus for tfidf testing
    tree = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/bc3_corpus/corpus_processed.xml')
    root_corpus = tree.getroot()
    threads = bc3.get_threads_in_root(root_corpus)

    # Import bc3_threads_words
    from data.helper_variables import bc3_threads_words

    # Calculate the vocab list for corpus + the email
    # Note: We use the cached vocab list in helper_variables.py!
    # This is needed for the tf-idf vectors, see function for more details
    from data.helper_variables import bc3_vocab_list
    bc3_vocab_list = set(bc3_vocab_list)
    email_words = bc3.get_words_in_email(email_message)
    email_vocab_list = set(email_words)
    all_vocab_list = bc3_vocab_list.union(email_vocab_list)
    all_vocab_list_index = {word: index for index, word in enumerate(all_vocab_list)}

    # Precalculate the tf-idf vectors and centroid vectors
    cached_tf_idf_vectors, cached_tf_local_idf_vectors = dict(), dict()
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence_words = bc3.get_words_in_sentence(sentence)

        tf_idf_vector = get_tf_idf_vector(sentence, email_message, email_words, 
                        sentence_words, bc3_threads_words, threads, all_vocab_list_index)
        tf_local_idf_vector = get_tf_local_idf_vector(sentence, email_message, 
                              email_words, sentence_words, all_vocab_list_index)
        cached_tf_idf_vectors[i] = tf_idf_vector
        cached_tf_local_idf_vectors[i] = tf_local_idf_vector
    centroid_vector = sum(cached_tf_idf_vectors.values()) / len(cached_tf_idf_vectors)
    local_centroid_vector = sum(cached_tf_local_idf_vectors.values()) / len(cached_tf_local_idf_vectors)

    sentence_vectors = []
    for i in range(len(sentences)):
        # Calculate some sentence metadata
        sentence = sentences[i]
        num_recipients = len(to_cc.split(','))
        tf_idf_vector = cached_tf_idf_vectors[i]
        tf_local_idf_vector = cached_tf_local_idf_vectors[i]

        # Calculate the sentence vector
        sentence_vector = bc3.vectorize_sentence(sentence, i, email_message, title,
                          num_recipients, sentences, 1, tf_idf_vector,
                          tf_local_idf_vector, centroid_vector, local_centroid_vector)
        sentence_vectors.append(sentence_vector)

    return sentence_vectors

def get_sents_by_importance(random_forest_full, sent_vectors, sentences, display_sentences):
    "Return the sentences sorted by importance score"

    sent_scores = [float(random_forest_full.predict(sent)) for sent in sent_vectors]

    sents_by_importance = [x for (y, x) in sorted(zip(sent_scores, display_sentences), reverse=True)]

    return sents_by_importance

def get_original_sent_order(sentences):
    "Return the position of each sentence in the email as a dictionary {sentence1: index1, sentence2: index2, ...}"

    original_sent_order = dict()
    count = 1 

    for sent in sentences:
        original_sent_order[sent] = count 
        count += 1

    return original_sent_order

def preprocess_email(email_message):
    '''Takes in an email, and outputs a cleaned version, where emails, long numbers, and URLs are replaced with a token.'''
    sentences = bc3.get_sentences_in_email(email_message)
    clean_sentences = []
    for sentence in sentences:
        clean_sentence = sentence
        clean_sentence = bc3.replace_emails_with_token(clean_sentence)
        clean_sentence = bc3.replace_longnumbers_with_token(clean_sentence)
        clean_sentence = bc3.replace_urls_with_token(clean_sentence)
        clean_sentences.append(clean_sentence)
    return ' '.join(clean_sentences)

def process_email(email_message, title, to_cc):
    "Output a list of sentences sorted by importance and a dictionary of the order of the sentences"

    # Get original sentences in email, as well as shortened sentences to display
    sentences = bc3.get_sentences_in_email(email_message)
    display_sentences = bc3.get_sentences_in_email(remove_adverb(email_message))

    # Dictionary that maps a shortened display sentence to the original
    display_sent_to_original = {display_sentences[i]: sentences[i] for i in range(len(sentences))}

    # Process the email message for vectorization and vectorize it
    processed_email_message = preprocess_email(email_message)
    sent_vectors = vectorize_email(processed_email_message, title, to_cc)

    # Calculate variables to return:
    # A list of sentences sorted by importance, and a dictionary of the order of the sentences
    random_forest_full = pickle_load("random_forest_full")
    sents_by_importance = get_sents_by_importance(random_forest_full, sent_vectors, sentences, display_sentences)
    original_sent_order = get_original_sent_order(sentences)

    return sents_by_importance, original_sent_order, display_sent_to_original

email_message = "My fellow students. A belated Happy New Year and warm wishes for the rest of the semester ahead. Our time here at NYU Shanghai is flying by swiftly and I am excited to announce we are approaching the election season of the 2015-2016 Student Government at NYU Shanghai. I would like to invite you to our Elections Informational Meeting on March 5th, during the 12:30-1:45 lunch hour (room TBD) for an extremely important and insightful chance to learn about the upcoming elections. At this meeting, the Student Government and the Elections Board will be present to provide information and answer questions about the election timeline, candidacy, and campaign rules and regulations. For those interested in running for an elected position, you are highly encouraged to attend for your own benefit. This year's elections will be notably different from previous elections as the current Executive Board has been working determinedly to revise the Student Constitution and structure of Student Government to reflect a more efficient,  inclusive, and precise organizational structure that will better suit the needs of the Student Body and Student Governments of the future. We invite you all to read through the 2015 Constitutional Revisions and note the changes from the Original Constitution. You will find the differences to be at once significant, but nuanced. The Student Constitution will be open to you for comment until March 5. After you review the Constitution, please vote to ratify the revisions here on OrgSync. If you have comments on the Constitution, please email shanghai.student.government@nyu.edu. Please RSVP here if you plan to attend the Elections Informational Meeting. Please see below to read the 2015 Constitutional Revisions. Student Constitution 2015-2016. If you wish to read the Original Constitution, see below. Student Government is the heart of student interests and activity at NYU Shanghai and the Student Constitution gives life to the Student Government. If you are passionate about enhancing the student experience and community spirit at NYU Shanghai, I encourage you to read the Constitution, understand it, and run for Student Government."
to_cc = "NYU Shanghai Students CO17 <nyushanghai-students-co17-group@nyu.edu>, NYU Shanghai Students CO18 <nyushanghai-students-co18-group@nyu.edu>"
title = "Constitution Revisions & Elections Info Meeting"

if __name__ == '__main__':
    # Test vectorizing an email and output the resulting vector
    title = "Constitution Revisions & Elections Info Meeting"
    to_cc = "NYU Shanghai Students CO17 <nyushanghai-students-co17-group@nyu.edu>, NYU Shanghai Students CO18 <nyushanghai-students-co18-group@nyu.edu>"
    email_message = "We are saddened by the thousands of lives loss in the Nepal Earthquake. We thank the NYU Shanghai community for showing its compassion and spirit in sending prayers and thoughts to those that have been impacted. We would like to bring the community together at 7pm Monday 5/4 in Room 808 for a candlelight vigil followed by a performance by the Other Octaves. We know the journey to rebuilding will take time and we stand committed to doing our part. To begin, the Rotaract Club at NYU Shanghai will be hosting a benefit event on Sunday May 10 from 530pm-7pm in conjunction with the Rotary of Shanghai. The benefit event will include talks and performances by members of the NYU Shanghai community with a special performance by the Symphony Orchestra. All proceeds will be donated to providing shelterboxes that will be sent to Nepal. You can reach out to Hunter Jarvis ('17) hmj239@nyu.edu and Lillian Korinek ('18) lhk250@nyu.edu if you would like to be involved."
    sents_by_importance, original_sent_order, display_sent_to_original = process_email(email_message, title, to_cc)

    print(sents_by_importance)
    print(original_sent_order)
    print(display_sent_to_original)
