#Ausawin Saehaan
#CMSC416
#04/16/2022
#Programming Assignment 5: Twitter Sentiment Analysis

#***Purpose***
#This is a separate program from sentiment.py, where sentiment.py's output is compared with the gold standard "key" data, which is named sentiment-test-key.txt 
#The accuracy of the sense tagged output created by sentiment.py is calculated here and a confusion matrix is generated as well

#***How to Use***
#Make sure to download and have all files in the same location (scorer.py, sentiment-test-key.txt, my-sentiment-answers.txt)
#Open command prompt (cmd) and change your cmd's directory/location to that of all your files using 'cd'

#Enter the command: python scorer.py my-sentiment-answers.txt sentiment-test-key.txt 

#***Example Output***
#Overall accuracy of tagging:...%
#Overall accuracy of baseline sentiment: ...%
#Confusion matrix

#***Algorithm***
#Read files sentiment-test-key.txt and my-sentiment-answers.txt as input
#Parse input generated from new files
#Add parsed input into an array/list
#Loop through list and check for matches between the golden standard key and output generated from wsd.py
#If number of matches increase, then accuracy increases; vice versa with decrease in correlation
#Take manipulated inputs and write to confusion matrix

from sys import argv
import re
import pandas as pd

#system commands called in cmd
argv[1] = "my-sentiment-answers.txt"
argv[2] = "sentiment-test-key.txt"

#read in sense tagged output and gold standard "key" data
f = open(argv[1], "r")
g = open(argv[2], "r")
fileMyAnswers = f.read()
fileKey = g.read()

#create lists for each file
i = re.split(r'\n', str(fileMyAnswers))
j = re.split(r'\n', str(fileKey))

#total amount of answer instances
answerTotal = 232

#use set intersection to get common answers between both lists
answerCorrect = list(set(i) & set(j))

#correct answers / total answers x 100 for a percentage
accuracy = len(answerCorrect)/answerTotal * 100

#most frequent sense baseline
baselinePositive = re.findall(r'sentiment=("positive")', str(fileKey))
answerBaseline = len(baselinePositive)/ answerTotal * 100 

#lists that will take input positives and negatives for confusion matrix
sentimentPredicted = []
sentimentActual = []

#appends correct and incorrect sentiments for matrix
index = 0
for index in range(len(i)):
    #if index in my-sentiment-answers.txt matches index in sentiment-test-key.txt
    if i[index] == j[index]:
        #if sentiment is positive for the matching answers of both files, then add positive to actual and prediction lists
        matchPositive = re.search(r'sentiment=("positive")',j[index])
        if matchPositive:
            sentimentActual.append('positive')
            sentimentPredicted.append('positive')
        #adds negative sentiment to both actual and prediction lists
        else:
            sentimentActual.append('negative')
            sentimentPredicted.append('negative')
    #if indices don't match and the sentiment is negative, adds positive to actual list but NEGATIVE to prediction list
    else:
        matchNegative = re.search(r'sentiment=("negative")',j[index])
        if matchNegative:
            sentimentActual.append('positive')
            sentimentPredicted.append('negative')
        else:
            sentimentActual.append('negative')
            sentimentPredicted.append('positive')


answerList = pd.Series(sentimentPredicted, name = 'Prediction')
keyList = pd.Series(sentimentActual, name = 'Actual')
confusionMatrix = pd.crosstab(answerList, keyList, margins = True)

#data = {'keyList': i, 'answerList': j}
#df = pd.DataFrame(data, columns=['keyList','answerList'])
#confusionMatrix = pd.crosstab(df['keyList'], df['answerList'], rownames=['Actual'],colnames=['Predicted'])

print("Overall accuracy of tagging: " + str(accuracy) + "%")
print("Overall accuracy of baseline sense: " + str(answerBaseline) + "%" + "\n")
print(confusionMatrix)







