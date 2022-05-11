#Ausawin Saehaan
#CMSC416
#03/27/2022
#Programming Assignment 4: Word Sense Disambiguation

#***Purpose***
#This program is used to detect the context behind specific words using a supervised learning approach.
#Given a manually annotated corpus, this was used as training data to teach a model to the program.
#The learned model will be used on another non-annotated test corpus to identify whether the word sense aligns more with phone or product.

#***How to Use***
#Make sure to download and have all files in the same location (wsd.py, line-test.txt, line-train.txt, my-model.txt, my-line-answers.txt)
    #my-model.txt and my-line-answers.txt can be new blank notepad files

#Make sure you have nltk installed via pip and nltk stopwords downloaded
    #   If unsure type pip install nltk into your cmd
    #   Also type 'nltk.download('stopwords') into Python's IDLE Shell
    
#Open command prompt (cmd) and change your cmd's directory/location to that of all your files using 'cd'
#Enter the command: python wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
#Open my-line-answers.txt to see word sense assigned to each instance

#***Example Output***
#Will consist of lines containing either phone or product for senseid
#...
#<answer instance="line-n.w8_106:13309:" senseid="phone"/>
#<answer instance="line-n.w8_119:16927:" senseid="product"/>
#...

#***Algorithm***
#Training file is read in for manipulation
    #Unnecessary characters are removed such as tags and stopwords to create a more consistent sense trainer
#After text is cleaned, the text is scanned(looped) through for matching regex
    #If "phone" is found to be contained within the matching regex then that match is added to the phone array
    #If "product" is found to be contained within the matching regex then that match is added to the product array
    #The purpose of adding certain text based off these keywords is to create a consistency for our word sense training
#Test file is then read in, to be used to check how consistent our word sense trainer is
#Similar tags and stopwords are removed for consistency
#A check is made where if words from the test file are also contained within either the phone or product list, then a count is incremented for either
    #The count that is greater will be the more consistent sense and the line id associated with the context will be printed out as an answer
    #These answers will be compared to within scorer.py which will contain the confusion matrix and baseline accuracy

#Decision list consisted of testing all words within an instance that surrounded the target word
#All words that surrounded the target word were associated with a particular sense

#Report from scorer.py:
#Overall accuracy of tagging: 95.23809523809523% 
#Overall accuracy of baseline sentiment: 57.14285714285714%

#Confusion Matrix:
#Actual      phone  product  All
#Prediction
#phone          68        4   72
#product         3       52   55
#All            71       56  127

from sys import argv
import re
import nltk
from nltk.corpus import stopwords

#system commands called in cmd
argv[1] = "line-train.txt"
argv[2] = "line-test.txt"
argv[3] = "my-model.txt"

#to be used to output the model the program learns from training file
#will display features and the sense it predicts, every two prints will correspond to one context (in order of test file)
h = open(argv[3],"w")

#will contain only answers displayed in format of key file
#j = open("my-line-answers.txt", "w")

#read line-train file as input
#parse and remove unneeded characters using re.sub()
f = open(argv[1], "r")
fileTrain = f.read()
#fileTrain = re.sub(r'\s+', "", fileTrain) #removes whitespaces
fileTrain = re.sub(r'<@>', "", fileTrain) #removes @ symbol
fileTrain = re.sub(r'(<|<\/)p>', "", fileTrain) #removes <p> and </p>
fileTrain = re.sub(r'(<|<\/)s>', "", fileTrain) #removes <s> and </s>
fileTrain = re.sub(r'(<|<\/)head>', "", fileTrain) #removes <head> and </head>

#filters out useless words according to nltk 
stopwordsTrain = set(stopwords.words('english'))
filtered_sentence = [w for w in fileTrain if not w.lower() in stopwordsTrain]
filtered_sentence = []
for w in fileTrain:
    if w not in stopwordsTrain:
        filtered_sentence.append(w)

#finds matches of "<answer instance...</instance>"
z = re.findall(r'<\s*answer.*?senseid\s*=\s*"(.*?)".*?>\s*<\s*context\s*>\s*(.*?)\s*<\s*\/context\s*>\s*<\s*\/\s*instance\s*>', str(fileTrain))

#loops through all regex matches, if one of the matches contains keyword "phone" add match to the phone list, repeat for keyword "product"
index = 0
sensePhone = []
senseProduct = []
senseDefault = 'phone'
for index in z:
    if "phone" in index:
        sensePhone.append(index)
    elif "product" in index:
        senseProduct.append(index)
    #else:
        #sensePhone.append(index) #if the index doesn't contain either phone or product, rely on default sense (phone)

sensePhone = re.split(r'\s+', str(sensePhone))
senseProduct = re.split(r'\s+', str(senseProduct))

#read line-test file as input
#parse and remove unneeded characters using re.sub()
g = open(argv[2], "r")
fileTest = g.read()
fileTest = re.sub(r'<@>', "", fileTest) #removes @ symbol
fileTest = re.sub(r'(<|<\/)p>',"", fileTest) #removes <p> and </p>
fileTest = re.sub(r'(<|<\/)s>',"", fileTest) #removes <s> and </s>
fileTest = re.sub(r'(<|<\/)head>',"", fileTest) #removes <head> and </head>

#filters out useless words according to nltk 
stopwordsTest = set(stopwords.words('english'))
filtered_sentence = [w for w in fileTest if not w.lower() in stopwordsTest]
filtered_sentence = []
for w in fileTest:
    if w not in stopwordsTest:
        filtered_sentence.append(w)

#use training data on test file, give each context instance a sense id based off consistency
#regex takes all characters and whitespaces from '<instance id...</instance>' within test file
a = re.findall(r'<\s*instance\s*(.*)>\s*<\s*context\s*>\s*(.*?)\s*<\s*\/context\s*>\s*<\s*\/\s*instance\s*>', str(fileTest))
#groups tuple elements together for each index
res = [' '.join(tups) for tups in a]
#regex takes all instances of '<instance id=line-n.w...:">' and adds them to a list
lineID = re.findall(r'<\s*instance.(.*)\s*>', str(fileTest))
#extracts position 4 to last position for each index in lineID (everything afterwards starting from 'l' in line-n...)
lineAnswers = [x[4:-1] for x in lineID] 

#lists that will contain instances' phone and product sense count and instance id
#all lists's indices will be relative to each position (first index of countList will correspond to first index of idList and senseList)
countList = []
idList = []
senseList = []

#loops through all contexts (126 contexts)
for index in res: 
    #counters that keep track of phone and product senses
    countPhone=0
    countProduct=0
    #within each context, split each word by whitespace, making the words to the left and right of the whitespace an index
    i = re.split(r'\s+', str(index))
    #loops through each word within a context
    for n in i: 
        #if a word within the test file is also contained within the phone or product sense list, up the counter(s)
        if n in sensePhone:
            countPhone += 1
        if n in senseProduct:
            countProduct += 1
    #if more words were related to phone then add the phone sense to the sense list, if more related to product then add product
    #if any other situation (such as equal amount of words related to phone and product) then default back to phone sense
    if countPhone > countProduct:
        senseList.append("phone")
    elif countPhone < countProduct:
        senseList.append("product")
    else:
        senseList.append("phone")
    #total counts of each context's word sense were added to a new list 
    countList.append((countPhone,countProduct))

#similar to loop above but uses range and len so that index increment could be used to access list position
for index in range(len(res)):
    #line-n instance IDs from test file are added to the id list
    idList.append((lineAnswers[index]))
    #results of decision list
    #print("(phone,product) count:" + str(countList[index]))
    print("<answer instance=\"" + str(idList[index]) + "\"" + " senseid=\"" + str(senseList[index]) + "\"/>")
    h.write("(phone,product) count:" + str(countList[index]) + "\n")
    h.write("<answer instance=\"" + str(idList[index]) + "\"" + " senseid=\"" + str(senseList[index]) + "\"/>" + "\n")
    #j.write("<answer instance=\"" + str(idList[index]) + "\"" + " senseid=\"" + str(senseList[index]) + "\"/>" + "\n")

#close files (output doesn't show up unless properly closed)
h.close()
#j.close()

