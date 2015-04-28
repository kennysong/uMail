import numpy as np
import xml.etree.ElementTree as ET
import string
import math
import re

import data.vocab_data
import vectorize_bc3 as v_bc3

from scipy.spatial import distance
from external_modules import inflect

def get_tf_idf_vector(sentence, email, threads, word_counts_per_thread, vocab_list_index):
    '''Takes in a sentence and metadata, and returns a tf-idf vector of the sentence words.
       idf = # of threads in BC3 corpus that have the word.'''

    # Clean up words in the email and the sentence
    email_words = v_bc3.clean_up_words(email.split())
    clean_sentence_words = v_bc3.clean_up_words(sentence.text.split())

    # Calculate term-frequency, number of times word appears in the email
    tf_vector = np.zeros(len(vocab_list_index))
    for word in clean_sentence_words:
        occurrences = word_counts_per_thread[thread][word]
        tf = occurrences / float(len(thread_words))
        word_index = vocab_list_index[word]
        tf_vector[word_index] = tf

    # Calculate inverse document frequency, on # of threads in entire corpus that have the word
    idf_vector = np.zeros(len(vocab_list_index))
    thread_words = [set(clean_up_words(get_words_in_thread(thread))) for thread in threads]
    for word in clean_sentence_words:
        num_threads_with_word = 0.0
        for thread_word_list in thread_words:
            if word in thread_word_list: num_threads_with_word += 1
        idf = math.log(len(threads) / num_threads_with_word)
        word_index = vocab_list_index[word]
        idf_vector[word_index] = idf

    # Get tf*idf vector
    tf_idf_vector = tf_vector * idf_vector

    return tf_idf_vector

def get_tf_local_idf_vector(sentence, thread, root, word_counts_per_thread, vocab_list_index):
    '''Takes in a sentence and metadata, and returns a tf-local-idf vector of the sentence words.
       local-idf = # of emails that the word appears in = 1.'''

    # Clean up words in the email and the sentence
    thread_words = clean_up_words(get_words_in_thread(thread))
    clean_sentence_words = clean_up_words(sentence.text.split())

    # Calculate term-frequency, number of times word appears in thread
    tf_vector = np.zeros(len(vocab_list_index)) # Global variable, see below
    for word in clean_sentence_words:
        occurrences = word_counts_per_thread[thread][word] # Global variable, see below
        tf = occurrences / float(len(thread_words))
        word_index = vocab_list_index[word]
        tf_vector[word_index] = tf

    # Calculate inverse document frequency, on # of emails in thread that have the word
    idf_vector = np.zeros(len(vocab_list_index))
    emails = get_emails_in_thread(thread)
    email_words = [set(clean_up_words(get_words_in_email(email))) for email in emails]
    for word in clean_sentence_words:
        num_emails_with_word = 0.0
        for email_word_list in email_words:
            if word in email_word_list: num_emails_with_word += 1
        idf = math.log(len(emails) / num_emails_with_word)
        word_index = vocab_list_index[word]
        idf_vector[word_index] = idf

    # Get tf*idf vector
    tf_idf_vector = tf_vector * idf_vector

    return tf_idf_vector

def vectorize_email(email_message, vocab_list, title, to_cc):
    '''Takes in an email and metadata, returns list of feature vectors for sentences.'''

    # Split email into list of sentences
    sentences = v_bc3.get_sentences_in_email(email_message)

    # Load the BC3 corpus for tfidf testing
    tree = ET.parse('bc3_corpus/bc3corpus.1.0/corpus.xml')
    root = tree.getroot()
    threads = v_bc3.get_threads_in_root(root)

    # Calculate the vocab list for corpus + the email
    # Note: We use the cached vocab list in data.vocab_data!
    # This is needed for the tf-idf vectors, see function for more details
    bc3_vocab_list = data.vocab_data.vocab_list
    email_vocab_list = v_bc3.clean_up_words(email_message.split())
    all_vocab_list = bc3_vocab_list + email_vocab_list
    all_vocab_list_index = {word: index for index, word in enumerate(all_vocab_list)}

    # Precalculate the tf-idf vectors and centroid vectors


    sentence_vectors = []
    for i in range(len(sentences)):
        # Calculate some sentence metadata
        sentence = sentences[i]
        num_recipients = len(to_cc.split(','))
        tf_idf_vector = cached_tf_idf_vectors[i]
        tf_local_idf_vector = cached_tf_local_idf_vectors[i]
        centroid_vector = cached_centroid_vectors[i]
        local_centroid_vector = cached_local_centroid_vectors[i]

        # Calculate the sentence vector
        sentence_vector = v_bc3.vectorize_sentence(sentence, i, emails, subject, 
                          num_recipients, sentences, 1, tf_idf_vector, 
                          tf_local_idf_vector, centroid_vector, local_centroid_vector)
        sentence_vectors.append(sentence_vector)

    return sentence_vectors

if __name__ == '__main__':
    title = "Constitution Revisions & Elections Info Meeting"
    to_cc = "NYU Shanghai Students CO17 <nyushanghai-students-co17-group@nyu.edu>, NYU Shanghai Students CO18 <nyushanghai-students-co18-group@nyu.edu>"
    email_message = "My fellow students. A belated Happy New Year and warm wishes for the rest of the semester ahead. Our time here at NYU Shanghai is flying by swiftly and I am excited to announce we are approaching the election season of the 2015-2016 Student Government at NYU Shanghai. I would like to invite you to our Elections Informational Meeting on March 5th, during the 12:30-1:45 lunch hour (room TBD) for an extremely important and insightful chance to learn about the upcoming elections. At this meeting, the Student Government and the Elections Board will be present to provide information and answer questions about the election timeline, candidacy, and campaign rules and regulations. For those interested in running for an elected position, you are highly encouraged to attend for your own benefit. This year's elections will be notably different from previous elections as the current Executive Board has been working determinedly to revise the Student Constitution and structure of Student Government to reflect a more efficient,  inclusive, and precise organizational structure that will better suit the needs of the Student Body and Student Governments of the future. We invite you all to read through the 2015 Constitutional Revisions and note the changes from the Original Constitution. You will find the differences to be at once significant, but nuanced. The Student Constitution will be open to you for comment until March. 5. After you review the Constitution, please vote to ratify the revisions here on OrgSync. If you have comments on the Constitution, please email shanghai.student.government@nyu.edu. Please RSVP here if you plan to attend the Elections Informational Meeting. Please see below to read the 2015 Constitutional Revisions. Student Constitution 2015-2016. If you wish to read the Original Constitution, see below. Student Government is the heart of student interests and activity at NYU Shanghai and the Student Constitution gives life to the Student Government. If you are passionate about enhancing the student experience and community spirit at NYU Shanghai, I encourage you to read the Constitution, understand it, and run for Student Government." 

    vector = vectorize_email(email_message, title, to_cc)
