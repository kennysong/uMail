import cPickle as pickle

#Get thread_listno of current thread

def thread_listno_of_current_thread(thread):
	for listno in thread:
		if listno.tag == "listno": 
			return listno.text

#pickle data into /data with file name: data_data.pck

def pickle_data(data, file_name = str):	

	data_location = "data/" + file_name + ".pck"

	f = open(data_location, "w")
	
	pickle.dump(data,f)

	f.close()
