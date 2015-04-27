import numpy as np
import xml.etree.ElementTree as ET
import string
import math

from data.vocab_data import *

from scipy.spatial import distance
from external_modules import inflect

def get_threads_in_root(root):
    return list(root)

def get_words_in_sentence(sentence):
    return sentence.text.split()

def get_emails_in_thread(thread):
    emails = []
    for i in range(len(thread)):
        if thread[i].tag == 'DOC':
            emails.append(thread[i])
    return emails

def get_sentences_in_email(email):
    text_tag = None
    for i in range(len(email)):
        if email[i].tag == 'Text':
            text_tag = email[i]
    return list(text_tag)

def get_words_in_thread(thread):
    words = []
    emails = get_emails_in_thread(thread)
    for email in emails:
        sentences = get_sentences_in_email(email)
        for sentence in sentences:
            words += get_words_in_sentence(sentence)
    return words

def clean_up_words(word_list):
    # Make lowercase
    clean_word_list = [word.lower() for word in word_list]

    # Remove non alphabet or numbers
    for i in range(len(clean_word_list)):
        clean_word_list[i] = ''.join([t for t in clean_word_list[i] if (t in '1234567890' or t in string.ascii_lowercase)])

    # Make singular
    inflect_engine = inflect.engine()
    for i in range(len(clean_word_list)):
        if inflect_engine.singular_noun(clean_word_list[i]) != False:
            clean_word_list[i] = inflect_engine.singular_noun(clean_word_list[i])

    return clean_word_list

def tfidf(word,email_message,threads,words_of_emails):

    #word is string
    #email_message is a list of sentence in the email

    words_in_thread = [] 
    for thread in threads:
        words_in_thread.append(get_words_in_thread(thread))


    tf = (float(words_of_emails.count(word)))/len(words_of_emails)

    no_of_email_with_t = 0 
    for item in words_in_thread:
        if word in item: no_of_email_with_t+=1

    if no_of_email_with_t != 0: idf = math.log(float(len(threads))/no_of_email_with_t)
    else: idf = 1 

    return (tf*idf)

def centroid_vector_tfidfvector(email_message,vocab_list,threads,words_of_emails):
    tfidf_allsentence_in_email = np.array([np.zeros(len(vocab_list)) for j in range(len(email_message))])

    for i in range(len(email_message)):
        words_of_sentence = []
        words_of_sentence.append(clean_up_words(email_message[i].split()))

        sentence_tfidf = []
        for word in words_of_sentence:
            sentence_tfidf.append(tfidf(word,email_message,threads,words_of_emails))

        for j in range(len(tfidf_allsentence_in_email[i])):
            if vocab_list[j] in words_of_sentence: tfidf_allsentence_in_email[i][j] = sentence_tfidf[sentence_tfidf.index(vocab_list[j])]

    centroid_vector = sum(tfidf_allsentence_in_email)/len(tfidf_allsentence_in_email)
    return tfidf_allsentence_in_email, centroid_vector 
    
def vectorize_sentence(email_message,sentence,vocab_list,title,to_Cc,tfidf_allsentence_in_email,centroid_vector):
    vector = []

    #Thread Line Number"
    vector.append(email_message.index(sentence))

    #Relative Pos. in thread 
    vector.append(float(email_message.index(sentence))/len(email_message))

    #print "centroid similarity"
    #Centroid similarity     #Centroid vector is the average of the tfidf vectors of all the sentences in the email 
    tf_idf_vector = tfidf_allsentence_in_email[email_message.index(sentence)]

    centroid_similarity = 1 - distance.cosine(tf_idf_vector, centroid_vector)
    vector.append(centroid_similarity)

    #print "Local centroid similarity" 
    vector.append(centroid_similarity)

    #length
    vector.append(len(sentence.split()))

    #TF*IDF Sum
    vector.append(sum(tf_idf_vector))

    #tf*idf average
    vector.append(sum(tf_idf_vector)/float(len(tf_idf_vector)))

    #Is Question
    if "?" in sentence: vector.append(1)
    else: vector.append(0 )

    #print "Email Number"
    vector.append(1)

    #Relative Pos. in email
    vector.append(float(email_message.index(sentence))/len(email_message))

    #subject similarity 
    title = clean_up_words(title.split())

    sentence_word_clean = clean_up_words(sentence.split())
    vector.append(len(set(title).intersection(sentence_word_clean)))

    #num recipients
    vector.append(to_Cc.count("@"))

    vector = np.array(vector)

    return vector 

def vectorize_email(email_message,vocab_list,title,to_Cc):
    email_message = email_message.split(".")

    vector = []

    tree = ET.parse('bc3_corpus/bc3corpus.1.0/corpus.xml')
    root = tree.getroot()
    threads = get_threads_in_root(root)

    sentences = []
    for sentence in email_message:
        sentences.append(sentence)

    words_of_emails = []
    for sentence in sentences:
        sentence_cleaned_up = clean_up_words(sentence.split())
        for word in sentence_cleaned_up:
            words_of_emails.append(word)
            
    tfidf_allsentence_in_email,centroid_vector = centroid_vector_tfidfvector(email_message,vocab_list,threads,words_of_emails)

    for sentence in email_message:
        vector.append(vectorize_sentence(email_message,sentence,vocab_list,title,to_Cc,tfidf_allsentence_in_email,centroid_vector))

    return vector 

to_Cc = "NYU Shanghai Students CO17 <nyushanghai-students-co17-group@nyu.edu>, NYU Shanghai Students CO18 <nyushanghai-students-co18-group@nyu.edu>"

email_message = "My fellow students. A belated Happy New Year and warm wishes for the rest of the semester ahead. Our time here at NYU Shanghai is flying by swiftly and I am excited to announce we are approaching the election season of the 2015-2016 Student Government at NYU Shanghai. I would like to invite you to our Elections Informational Meeting on March 5th, during the 12:30-1:45 lunch hour (room TBD) for an extremely important and insightful chance to learn about the upcoming elections. At this meeting, the Student Government and the Elections Board will be present to provide information and answer questions about the election timeline, candidacy, and campaign rules and regulations. For those interested in running for an elected position, you are highly encouraged to attend for your own benefit. This year's elections will be notably different from previous elections as the current Executive Board has been working determinedly to revise the Student Constitution and structure of Student Government to reflect a more efficient,  inclusive, and precise organizational structure that will better suit the needs of the Student Body and Student Governments of the future. We invite you all to read through the 2015 Constitutional Revisions and note the changes from the Original Constitution. You will find the differences to be at once significant, but nuanced. The Student Constitution will be open to you for comment until March. 5. After you review the Constitution, please vote to ratify the revisions here on OrgSync. If you have comments on the Constitution, please email shanghai.student.government@nyu.edu. Please RSVP here if you plan to attend the Elections Informational Meeting. Please see below to read the 2015 Constitutional Revisions. Student Constitution 2015-2016. If you wish to read the Original Constitution, see below. Student Government is the heart of student interests and activity at NYU Shanghai and the Student Constitution gives life to the Student Government. If you are passionate about enhancing the student experience and community spirit at NYU Shanghai, I encourage you to read the Constitution, understand it, and run for Student Government." #A long string containing all the sentence in the email body

title = "Constitution Revisions & Elections Info Meeting"

vector = vectorize_email(email_message,vocab_list,title,to_Cc)
