##S Some code that we used for a one-off calculation

## Calculate and cache all tf_idf vectors, in {sentence: tf-idf vector}
cached_tf_idf_vectors, cached_tf_local_idf_vectors = dict(), dict()

# For distributing this on multiple computers
thread_partitions = [(0, 15), (15, 30), (30, 40)]
partition_num = 0

for thread_index in range(len(root)):
        # For distributing this on multiple computers
        partition = thread_partitions[partition_num]
        if not (partition[0] <= thread_index < partition[1]):
                continue

        # Get tf-idf vector for each sentence in this thread
        thread = root[thread_index]
        thread_sentences = get_sentences_in_thread(thread)
        for sentence in thread_sentences:
                # Get the key for the sentence as thread_id-sentence_id
                thread_id = thread[1].text
                sentence_id = sentence.attrib['id']
                sentence_key = thread_id + '-' + sentence_id

                tf_idf_vector = get_tf_idf_vector(sentence, thread, root)
                cached_tf_idf_vectors[sentence_key] = tf_idf_vector
                tf_local_idf_vector = get_tf_local_idf_vector(sentence, thread, root)
                cached_tf_local_idf_vectors[sentence_key] = tf_local_idf_vector
        
        print("Calculated tf-idf for thread %i" % thread_index)

print("Finished pre-calculating tf-idf stuff")

## Pickle the tf-idf dictionaries
tf_idf_file = open("pickled_data/cached_tf_idf_vectors" + str(partition_num + 1), "wb")
pickle.dump(cached_tf_idf_vectors, tf_idf_file, protocol=2)
tf_idf_file.close()
tf_local_idf_file = open("pickled_data/cached_tf_local_idf_vectors" + str(partition_num + 1), "wb")
pickle.dump(cached_tf_local_idf_vectors, tf_local_idf_file, protocol=2)
tf_local_idf_file.close()

print("Successfully pickled")

## Combine tf_idf vectors from pickled cache into one pickle file
tf_idf_file1 = open("pickled_data/cached_tf_idf_vectors1", "rb")
tf_idf_file2 = open("pickled_data/cached_tf_idf_vectors2", "rb")
tf_idf_file3 = open("pickled_data/cached_tf_idf_vectors3", "rb")
tf_local_idf_file1 = open("pickled_data/cached_tf_local_idf_vectors1", "rb")
tf_local_idf_file2 = open("pickled_data/cached_tf_local_idf_vectors2", "rb")
tf_local_idf_file3 = open("pickled_data/cached_tf_local_idf_vectors3", "rb")

tf_idf_cache1 = pickle.load(tf_idf_file1)
tf_idf_cache2 = pickle.load(tf_idf_file2)
tf_idf_cache3 = pickle.load(tf_idf_file3)
tf_local_idf_cache1 = pickle.load(tf_local_idf_file1)
tf_local_idf_cache2 = pickle.load(tf_local_idf_file2)
tf_local_idf_cache3 = pickle.load(tf_local_idf_file3)

tf_idf_cache1.update(tf_idf_cache2)
tf_idf_cache1.update(tf_idf_cache3)
tf_local_idf_cache1.update(tf_local_idf_cache2)
tf_local_idf_cache1.update(tf_local_idf_cache3)

tf_idf_file_all = open("pickled_data/cached_tf_idf_vectors_all", "wb")
pickle.dump(tf_idf_cache1, tf_idf_file_all, protocol=2)
tf_idf_file_all.close()
tf_local_idf_file_all = open("pickled_data/cached_tf_local_idf_vectors_all", "wb")
pickle.dump(tf_local_idf_cache1, tf_local_idf_file_all, protocol=2)
tf_local_idf_file_all.close()