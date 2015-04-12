import numpy as np
import xml.etree.ElementTree as ET
import string
import math
import collections
import copy
import pickle

from scipy.spatial import distance
from external_modules import inflect

def get_threads_in_root(root):
        return list(root)

def get_emails_in_thread(thread):
        emails = []
        for i in range(len(thread)):
                if thread[i].tag == 'DOC':
                        emails.append(thread[i])
        return emails

def get_email_of_sentence(sentence, emails):
        for email in emails:
                sentences = get_sentences_in_email(email)
                if sentence in sentences:
                        return email

def get_sentences_in_thread(thread):
        emails = get_emails_in_thread(thread)
        thread_sentences = []
        for email in emails:
                sentences = get_sentences_in_email(email)
                for sentence in sentences:
                        thread_sentences.append(sentence)

        return thread_sentences

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

def get_words_in_email(email):
        word_list = []
        sentences = get_sentences_in_email(email)
        for sentence in sentences:
                word_list += get_words_in_sentence(sentence)
        return word_list

def get_words_in_sentence(sentence):
        return sentence.text.split()

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

def get_num_recipients(email):
        num_recipients = 0
        for email_tag in email:
                if email_tag.tag == 'To' or email_tag.tag == 'Cc':
                        if email_tag.text is None: continue
                        num_recipients += email_tag.text.count('@')
        return num_recipients

def get_relative_position_in_email(sentence, emails):
        for email in emails:
                sentences = get_sentences_in_email(email)
                if sentence in sentences:
                        return sentences.index(sentence) / float(len(sentences)) * 100

def get_subject_similarity(subject, sentence):
        # Split into lists of words
        subject_words = clean_up_words(subject.text.split())
        sentence_words = clean_up_words(sentence.text.split())

        # Get # of words occurring in both
        overlap = len(set(subject_words).intersection(set(sentence_words)))
        return overlap

def get_tf_idf_vector(sentence, thread, root, word_counts_per_thread, vocab_list_index):
        threads = get_threads_in_root(root)
        thread_words = clean_up_words(get_words_in_thread(thread))
        clean_sentence = clean_up_words(sentence.text.split())

        # Calculate term-frequency, number of times word appears in thread
        tf_vector = np.zeros(len(vocab_list_index)) # Global variable, see below
        for word in clean_sentence:
                occurrences = word_counts_per_thread[thread][word] # Global variable, see below
                tf = occurrences / float(len(thread_words))
                word_index = vocab_list_index[word]
                tf_vector[word_index] = tf

        # Calculate inverse document frequency, on # of threads in entire corpus that have the word
        idf_vector = np.zeros(len(vocab_list_index))
        thread_words = [set(clean_up_words(get_words_in_thread(thread))) for thread in threads]
        for word in clean_sentence:
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
        thread_words = clean_up_words(get_words_in_thread(thread))
        clean_sentence = clean_up_words(sentence.text.split())

        # Calculate term-frequency, number of times word appears in thread
        tf_vector = np.zeros(len(vocab_list_index)) # Global variable, see below
        for word in clean_sentence:
                occurrences = word_counts_per_thread[thread][word] # Global variable, see below
                tf = occurrences / float(len(thread_words))
                word_index = vocab_list_index[word]
                tf_vector[word_index] = tf

        # Calculate inverse document frequency, on # of emails in thread that have the word
        idf_vector = np.zeros(len(vocab_list_index))
        emails = get_emails_in_thread(thread)
        email_words = [set(clean_up_words(get_words_in_email(email))) for email in emails]
        for word in clean_sentence:
                num_emails_with_word = 0.0
                for email_word_list in email_words:
                        if word in email_word_list: num_emails_with_word += 1
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

def get_sentenceID_from_sentences(sentence): #in string form 
        sentenceID = []
        for item in sentence:
                sentenceID.append(item.attrib["id"])
        return sentenceID

def get_sentenceID_from_thread(thread): #in string form 
        sentenceID = []
        annotations = get_annotation_from_thread(thread)
        sentences = []

        for annotation in annotations:
                sentences.append(get_sentences_from_annotations(annotation))

        for sentence in sentences:
                sentenceID.append(get_sentenceID_from_sentences(sentence[0]))

        sentenceID = [sentenceID[i][j] for i in range(len(sentenceID)) for j in range(len(sentenceID[i]))]

        return sentenceID

def normalize(score):
        tree_corpus = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
        root_corpus = tree_corpus.getroot() 

        corpus_sentence = dict() #a dictionary, each each is a dictionary of keys,values being sentence ID and sentence string respectivelsenten 

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
        tree_annotation = ET.parse('bc3_corpus/bc3corpus.1.0/annotation.xml')
        root_annotation = tree_annotation.getroot()

        tree_corpus = ET.parse("bc3_corpus/bc3corpus.1.0/corpus.xml")
        root_corpus = tree_corpus.getroot()

        #Creating a dictionary of thread number, each of which is another dictionary of the ID of each sentences in a thread
        corpus = dict()

        for thread in root_corpus:
                sentence_id = dict()
                sentences = get_sentences_in_thread(thread)

                for sentence in sentences: 
                        sentence_id[sentence.attrib['id']] = 0

                corpus[thread[1].text] = sentence_id 

        score = copy.deepcopy(corpus)

        for thread in root_annotation:
                sentenceID = get_sentenceID_from_thread(thread) #Getting all the sentences in the 3 summaries in annotation file 

                sentenceID_in_thread = score[thread[0].text].keys()

                for sentence in sentenceID_in_thread:
                        score[thread[0].text][sentence] = sentenceID.count(sentence)

        score = normalize(score)

        return score

def vectorize_training_data():
        ## Set up XML tree
        tree = ET.parse('bc3_corpus/bc3corpus.1.0/corpus.xml')
        root = tree.getroot()

        ## Get word occurrence counts for each thread
        word_counts_per_thread = dict()
        for thread in root:
                words = get_words_in_thread(thread)
                word_counts = collections.Counter(words)
                word_counts_per_thread[thread] = word_counts
        print("Finished getting word counts for each thread")

        ## Calculate vocab list for entire corpus
        vocab_list, vocab_list_set = [], set()
        for thread in root:
                thread_vocab_list = clean_up_words(get_words_in_thread(thread))
                for item in thread_vocab_list:
                        if item not in vocab_list_set: 
                                vocab_list.append(item)
                                vocab_list_set.add(item)
        vocab_list_index = {word: index for index, word in enumerate(vocab_list)}

        print("Finished constructing vocab list")

        ## Load tf_idf vector from pickled cache {thread_id-sentence_id: tf_idf vector}
        tf_idf_file = open("pickled_data/cached_tf_idf_vectors_all", "rb")
        tf_local_idf_file = open("pickled_data/cached_tf_local_idf_vectors_all", "rb")
        cached_tf_idf_vectors = pickle.load(tf_idf_file)
        cached_tf_local_idf_vectors = pickle.load(tf_local_idf_file)
        tf_idf_file.close()
        tf_local_idf_file.close()

        print("Loaded tf_idf vectors")

        ## Calculate and cache all centroid vectors, in {thread: centroid_vector}
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

        ## Traverse the XML tree, vectorizing each sentence as we encounter it
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
                        sentence = thread_sentences[i]
                        current_email = get_email_of_sentence(sentence, emails)
                        sentence_key = thread[1].text + '-' + sentence.attrib['id']

                        # Calculate all the (R14) features
                        thread_line_number = i
                        rel_position_in_thread = i / float(len(thread_sentences)) * 100
                        length = len(sentence.text.split())
                        is_question = 1 if '?' in sentence.text else 0
                        email_number = int(sentence.attrib['id'].split('.')[0])
                        rel_position_in_email = get_relative_position_in_email(sentence, emails)
                        subject_similarity = get_subject_similarity(subject, sentence)
                        num_recipients = get_num_recipients(current_email)
                        
                        tf_idf_vector = cached_tf_idf_vectors[sentence_key]
                        tf_idf_sum = sum(tf_idf_vector)
                        tf_idf_avg = sum(tf_idf_vector) / len(tf_idf_vector)
                        tf_local_idf_vector = cached_tf_local_idf_vectors[sentence_key]

                        centroid_vector = cached_centroid_vectors[thread]
                        local_centroid_vector = cached_local_centroid_vectors[thread]

                        centroid_similarity = 1 - distance.cosine(tf_idf_vector, centroid_vector)
                        local_centroid_similarity = 1 - distance.cosine(tf_local_idf_vector, local_centroid_vector)

                        # Finally add this vector to our global list
                        sentence_vector = np.array([thread_line_number, rel_position_in_thread, centroid_similarity, 
                                local_centroid_similarity, length, tf_idf_sum, tf_idf_avg, is_question,
                                email_number, rel_position_in_email, subject_similarity, num_recipients])
                        sentence_vectors.append(sentence_vector)
                        sentence_vectors_metadata.append((sentence_vector, {'id': sentence.attrib['id'], 'listno': thread_id}))

                print("Vectorized a thread")

        return sentence_vectors, sentence_vectors_metadata

if __name__ == '__main__':
        sentence_vectors, sentence_vectors_metadata = vectorize_training_data()