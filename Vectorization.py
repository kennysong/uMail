#Vectorize BC3 Corpus 

import fileinput
import numpy as np
import xml.etree.ElementTree as ET

#BeginPiece: parse XML file and paste data into a list
#the list[i] has the structure: [Name_of_ith_email,listno,[each line for each email],[each line for each email], ...]


tree = ET.parse('corpus_sample.xml')
root = tree.getroot()

Corpus = [[] for i in range(len(root))]

for i in range(len(root)):
        Corpus[i].append(root[i][0].text)
        Corpus[i].append(root[i][1].text)
        for j in range(2,len(root[i])):
                Doc_temp = []
                for k in range(len(root[i][j])):
                        Doc_temp.append(root[i][j][k].text)
                Corpus[i].append(Doc_temp)
        
#EndPiece


