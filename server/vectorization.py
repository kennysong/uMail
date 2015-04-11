#Vectorize BC3 Corpus 

import fileinput
import numpy as np
import xml.etree.ElementTree as ET
import string
from external_modules import inflect

#BeginPiece: parse XML file and paste data into a list
#the list[i] has the structure: [Name_of_ith_email,listno,[Received,from,To,Subject],[each line for each email],[each line for each email], ...]


tree = ET.parse('bc3_corpus/bc3corpus.1.0/corpus_sample.xml')
root = tree.getroot()
p = inflect.engine()
Corpus = [[] for i in range(len(root))]

for i in range(len(root)):
        Corpus[i].append(root[i][0].text)
        Corpus[i].append(root[i][1].text)
        for j in range(2,len(root[i])):
                doc_temp = []
                for k in range(4):
                        doc_temp.append(root[i][j][k].text)
                Corpus[i].append(doc_temp)

                text_temp = []                        
                for k in range(len(root[i][j][4])):
                        text_temp.append(root[i][j][4][k].text)
                Corpus[i].append(text_temp)
        
#EndPiece

#BeginPiece: Turn email into vectors

def SentenceFeature(root,i,j,k,l):
        #input a <Sent> </Sent>, meaning a sentence, generate the feature vector for the sentence
        #Input must be in the form of root[i][j][k][l] with l being the position of the email in the thread
        feature = []

        #Text Features include:
                #Thread Line Number
                #Relative Pos. in Thread
                #Centroid Similarity
                #Local Centroid Similarity
                #Length
                #TF*IDF Sum
                #TF*IDF Average
                #Is Question

        #thread line number
        feature.append[l+1]

        #Relave Pos. in thread
        feature.append[(l+1)/len(k)*100]

        #Centroid similarity

        #Local centroid similarity

        #Length
        sentence = root[i][j][k][l].text
        sentence_length = len(sentence.split())
        feature.append(sentence_length)

        #TF*IDF Sum

        #TF*IDF Average

        #Is question
        if "?" in sentence:
                feature.append("IsQuestion")
        else:
                feature.append("IsNotQuestion")
                
        #Donewith text feature, moving on to email feature

        #Email feature includes:
                #Email number
                #Relative Pos. in email
                #Subject similarity
                #Num_replies is not included as a feature (beginning of page 36)
                #Num_recipients
                #IsHiden is not included as a feature (end of page 34) 

        #Email number
        feature.append(j)

        #Relative Pos. in email
        feature.append((j+1)/(len(root[i])-2)*100)

        #Subject Similarity
        all_sentence = [] #include all sentence in the email
        for t in range(len(root[i][j][k])):
                all_sentence.append(root[i][j][k][t].text)

        all_sentence = [all_sentence[t].split() for t in range(len(all_sentence))]
        all_sentence = [m for n in all_sentence for m in n]
        print all_sentence
        all_sentence = [all_sentence[m].lower() for m in range(len(all_sentence))]

        subject_line = root[i][j][3].text
        subject_line = subject_line.lower()
        subject_line = subject_line.split()

        for i in range(len(all_sentence)):
                all_sentence[i] = ''.join([t for t in all_sentence[i] if (t in '1234567890' or t in string.ascii_lowercase)])
                if p.singular_noun(all_sentence[i]) != False:
                        all_sentence[i] = p.singular_noun(all_sentence[i])

        for i in range(len(subject_line)):
                subject_line[i] = ''.join([t for t in subject_line[i] if (t in '1234567890' or t in string.ascii_lowercase)])
                if p.singular_noun(subject_line[i]) != False:
                        subject_line[i] = p.singular_noun(subject_line[i])
                
        count = 0

        for item in all_sentence:
                if item in subject_line:
                        count += 1
                
        feature.append(count)

        #Num_recipients
        recipients = root[i][j][1].text + root[i][j][2].text

        count = 0

        for i in recipients:
                if "@" in i: count += 1

        feature.append(count)

        #
