#Ausawin Saehaan
#CMSC416
#04/16/2022
#Programming Assignment 5: Twitter Sentiment Analysis

#***Purpose***
#This program is used to detect the sentiment behind tweets using a supervised learning approach. Given a manually annotated corpus, this was used as training data to teach a model
#to the program. The learned model will be used on another non-annotated test corpus to identify positive or negative sentiments.

#***How to Use***
#Make sure to download and have all files in the same location (sentiment.py, sentiment-test.txt, sentiment-train.txt, my-model.txt, my-sentiment-answers.txt)
    #my-model.txt and my-sentiment-answers.txt can be new blank notepad files

#Make sure you have nltk installed via pip and nltk stopwords downloaded
    #   If unsure type pip install nltk into your cmd
    #   Also type 'nltk.download('stopwords') into Python's IDLE Shell

#Open command prompt (cmd) and change your cmd's directory/location to that of all your files using 'cd'
#Enter the command: python sentiment.py sentiment-train.txt sentiment-test.txt my-model.txt > my-sentiment-answers.txt
#Open my-sentiment-answers.txt to see sentiment assigned to each instance

#***Example Output***
##Will consist of lines containing either positive or negative sentiment
#...
#<answer instance="620979391984566272" sentiment="negative"/>
#<answer instance="621340584804888578" sentiment="positive"/>
#...

#***Algorithm***
#Training file is read in for manipulation
    #Unnecessary characters are removed or replaced to create a more consistent sentiment trainer
        #e.g. all @... such as @Microsoft is replaced with 'username', http links are replaced with url, emoticons are replaced with their word counterpart, hashtags are removed 
#After text file is cleaned, we find all instances and their contexts using re.findall()
    #Using the list generated from re.findall(), we loop through each index and add their contexts to new lists depending on whether they contain keywords 'positive' or 'negative'
    #These new lists become the bag of words for positive and negative sentiments
#Test file is then read in, to be used to check how consistent our word sense trainer is

#Accuracy of most frequent sense baseline: 68.96551724137932%
    
#Accuracy of tagging with no extra features                                                                                    - 76.72413793103449%
#Accuracy of tagging with feature 1: all words starting with uppercase removed                                                 - 71.55172413793103%
#Accuracy of tagging with feature 1+2: all words starting with uppercase removed + no punctuation for all instances            - 71.12068965517241%
#Accuracy of tagging with feature 1+2+3: all words starting with uppercase removed + no punctuation for all instances + no I's - 70.25862068965517%   


#Confusion Matrix no extra features:
#Actual      negative  positive  All
#Prediction
#negative          19        54   73
#positive           9       151  160
#All               28       205  233

#Confusion Matrix + feature 1
#Actual      negative  positive  All
#Prediction
#negative          21        52   73
#positive          15       145  160
#All               36       197  233

#Confusion Matrix + feature 1,2
#Actual      negative  positive  All
#Prediction
#negative          21        52   73
#positive          16       144  160
#All               37       196  233

#Confusion Matrix + feature 1,2,3
#Actual      negative  positive  All
#Prediction
#negative          23        50   73
#positive          20       140  160
#All               43       190  233

from sys import argv
import re
import nltk
from nltk.corpus import stopwords

#system commands called in cmd
argv[1] = "sentiment-train.txt"
argv[2] = "sentiment-test.txt"
argv[3] = "my-model.txt"

#to be used to output the model the program learns from training file
#will display features and the sense it predicts, every two prints will correspond to one context (in order of test file)
h = open(argv[3],"w")  

#read sentiment-train file as input
#normalize tweets using re.sub()
f = open(argv[1], "r")
fileTrain = f.read()
fileTrain = re.sub(r'@([a-zA-Z0-9]+)', "username", fileTrain) #substitute @.. instances from tweets with word 'username'
fileTrain = re.sub(r'http[^\s]*', "url", fileTrain) #substitute url links from tweets with word 'url'
fileTrain = re.sub(r':-?\)',"happy", fileTrain) #substitute happy face emoji with word 'happy'
fileTrain = re.sub(r':-?\(',"sad", fileTrain) #substitute happy face emoji with word 'happy'
fileTrain = re.sub(r'#','', fileTrain) #substitute #.. instances from tweets with their regular word counterpart

#filters out useless words according to nltk 
stopwordsTrain = set(stopwords.words('english'))
filtered_sentence = [w for w in fileTrain if not w.lower() in stopwordsTrain]
filtered_sentence = []
for w in fileTrain:
    if w not in stopwordsTrain:
        filtered_sentence.append(w)

#finds matches of "<answer instance...</instance>"
z = re.findall(r'<\s*answer.*?sentiment\s*=\s*"(.*?)".*?>\s*<\s*context\s*>\s*(.*?)\s*<\s*\/context\s*>\s*<\s*\/\s*instance\s*>', str(fileTrain))

#loops through all regex matches, if one of the matches contains keyword "phone" add match to the phone list, repeat for keyword "product"
index = 0
sentimentPositive = []
sentimentNegative = []

for index in z:
    if "positive" in index:
        sentimentPositive.append(index)
    elif "negative" in index:
        sentimentNegative.append(index)

#separates every word by whitespace
sentimentPositive = re.split(r'\s+', str(sentimentPositive))
sentimentNegative = re.split(r'\s+', str(sentimentNegative))

#additional features:
#Feature 1: instances contain no words that start with uppercase
sentimentTest1a = sentimentPositive
output1a = []
for sentence in sentimentTest1a:
    output1a.append(" ".join([word for word in sentence.strip().split(" ") if not re.match(r"[A-Z]",word)]))

sentimentTest1b = sentimentNegative
output1b = []
for sentence in sentimentTest1b:
    output1b.append(" ".join([word for word in sentence.strip().split(" ") if not re.match(r"[A-Z]",word)]))

#Feature 2: no punctuation for any instances
sentimentTest2a = output1a
output2a = []
for sentence in sentimentTest2a:
    output2a.append(" ".join([word for word in sentence.strip().split(" ") if not re.match(r"[^\w\s]",word)]))

sentimentTest2b = output1b
output2b = []
for sentence in sentimentTest2b:
    output2b.append(" ".join([word for word in sentence.strip().split(" ") if not re.match(r"[^\w\s]",word)]))

#Feature 3: no I's
sentimentTest3a = output2a
output3a = []
for sentence in sentimentTest2a:
    output3a.append(" ".join([word for word in sentence.strip().split(" ") if not re.match(r"[^\w\s]",word)]))

sentimentTest3b = output2b
output3b = []
for sentence in sentimentTest2b:
    output3b.append(" ".join([word for word in sentence.strip().split(" ") if not re.match(r"I",word)]))


#read sentiment-test file as input
#normalize tweets using re.sub()
g = open(argv[2], "r")
fileTest = g.read()
fileTest = re.sub(r'@([a-zA-Z0-9]+)', "username", fileTest) #substitute @.. instances from tweets with word 'username'
fileTest = re.sub(r'http[^\s]*', "url", fileTest) #substitute url links from tweets with word 'url'
fileTest = re.sub(r':-?\)',"happy", fileTest) #substitute happy face emoji with word 'happy'
fileTest = re.sub(r':-?\(',"sad", fileTest) #substitute happy face emoji with word 'happy'
fileTest = re.sub(r'#','', fileTest) #substitute #.. instances from tweets with their regular word counterpart

#filters out useless words according to nltk 
stopwordsTest = set(stopwords.words('english'))
filtered_sentence = [w for w in fileTest if not w.lower() in stopwordsTest]
filtered_sentence = []
for w in fileTest:
    if w not in stopwordsTest:
        filtered_sentence.append(w)

#use training data on test file, give each context instance a sentiment based off consistency
#regex takes all characters and whitespaces from '<instance id...</instance>' within test file
a = re.findall(r'\s*instance\s*(.*)>\s*<\s*context\s*>\s*(.*?)\s*<\s*\/context\s*>\s*<\s*\/\s*instance\s*', str(fileTest))

#groups tuple elements together for each index
res = [' '.join(tups) for tups in a]
#regex takes all instances of '<instance id=line-n.w...:">' and adds them to a list
lineID = re.findall(r'<\s*instance.(.*)\s*>', str(fileTest))
#extracts position 4 to last position for each index in lineID (everything afterwards starting from 'l' in line-n...)
lineAnswers = [x[4:-1] for x in lineID] 

#lists that will contain instances' positive and negative sentiment counts and instance id
#all lists's indices will be relative to each position (first index of countList will correspond to first index of idList and senseList)
countList = []
idList = []
sentimentList = []
probabilityList = []

#loops through all contexts 
for index in res: 
    #counters that keep track of positive and negative sentiment
    countPositive=0
    countNegative=0
    #within each context, split each word by whitespace, making the words to the left and right of the whitespace an index
    i = re.split(r'\s+', str(index))
    #loops through each word within a context
    for n in i: 
        #if a word within the test file is also contained within the positive or negative sentiment list, up the counter(s)
        if n in output3a:
            countPositive += 1
        if n in output3b:
            countNegative += 1
    #if more words were related to positive then add the positive sentiment to the sentiment list, if more related to negative then add negative
    #if any other situation (such as equal amount of words related to phone and product) then default back to phone sense
    if countPositive >  countNegative:
        sentimentList.append("positive")
    elif countPositive <  countNegative:
        sentimentList.append("negative")
    else:
        sentimentList.append("positive")
        
    #total counts of each context's word sense were added to a new list 
    countList.append((countPositive, countNegative))
    #probability/sentiment score is calculated by taking count of positive and negative words in an instance and taking ratio of difference between both counts over total word count of each instance 
    probability = (countPositive - countNegative) / len(i)
    probabilityList.append(probability)
    
#similar to loop above but uses range and len so that index increment could be used to access list position
for index in range(len(res)):
    #line-n instance IDs from test file are added to the id list
    idList.append((lineAnswers[index]))
    #results of decision list
    print("<answer instance=\"" + str(idList[index]) + "\"" + " sentiment=\"" + str(sentimentList[index]) + "\"/>") #goes to stdout (my-sentiment-answers.txt)
    h.write("(positive,negative) count:" + str(countList[index]) + "\t" + "probability score:" + str(probabilityList[index]) + "\n"
            "<answer instance=\"" + str(idList[index]) + "\"" + " sentiment=\"" + str(sentimentList[index]) + "\"/>" + "\n\n")

#close files (output doesn't show up unless properly closed)
h.close()
