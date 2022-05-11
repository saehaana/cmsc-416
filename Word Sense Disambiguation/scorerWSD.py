#Ausawin Saehaan
#CMSC416
#03/27/2022
#Programming Assignment 4: Word Sense Disambiguation

#***Purpose***
#This is a separate program from wsd.py, where wsd.py's output is compared with the gold standard "key" data, which is named line-key.txt 
#The accuracy of the sense tagged output created by wsd.py is calculated here and a confusion matrix is generated as well

#***How to Use***
#Make sure to download and have all files in the same location (wsd.py, scorer.py, line-test.txt, line-train.txt, line-key.txt)
#Open command prompt (cmd) and change your cmd's directory/location to that of all your files using 'cd'
#Enter the command: python scorer.py my-line-answers.txt line-key.txt 

#***Example Output***
#Overall accuracy of tagging: 95.23809523809523%
#Overall accuracy of baseline sentiment: 57.14285714285714%
#Confusion matrix

#***Algorithm***
#Read files line-key.txt and my-line-answers.txt as input
#Parse input generated from new files
#Add parsed input into an array/list
#Loop through list and check for matches between the golden standard key and output generated from wsd.py
#If number of matches increase, then accuracy increases; vice versa with decrease in correlation
#Take manipulated inputs and write to confusion matrix

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
import pandas as pd

#system commands called in cmd
argv[1] = "my-line-answers.txt"
argv[2] = "line-key.txt"

#read in sense tagged output and gold standard "key" data
f = open(argv[1], "r")
g = open(argv[2], "r")
fileMyAnswers = f.read()
fileKey = g.read()

i = re.split(r'\n', str(fileMyAnswers))
j = re.split(r'\n', str(fileKey))

#total amount of answer instances
answerTotal = 126

#use set intersection to get common answers between both lists 
answerCorrect = list(set(i) & set(j))

#correct answers / total answers x 100 for a percentage
accuracy = len(answerCorrect)/answerTotal * 100

#most frequent sense baseline
baselinePhone = re.findall(r'senseid=("phone")', str(fileKey))
answerBaseline = len(baselinePhone)/ answerTotal * 100 

#lists that will take input phone and product for confusion matrix
sensePredicted = []
senseActual = []

#appends correct and incorrect sentiments for matrix
index = 0
for index in range(len(i)):
    #if index in my-sentiment-answers.txt matches index in sentiment-test-key.txt
    if i[index] == j[index]:
        #if sense is phone for the matching answers of both files, then add phone to actual and prediction lists
        matchPhone = re.search(r'senseid=("phone")',j[index])
        if matchPhone:
            senseActual.append('phone')
            sensePredicted.append('phone')
        #adds product senseid to both actual and prediction lists
        else:
            senseActual.append('product')
            sensePredicted.append('product')
    #if indices don't match and the sense is product, adds phone to actual list but PRODUCT to prediction list
    else:
        matchProduct = re.search(r'senseid=("product")',j[index])
        if matchProduct:
            senseActual.append('phone')
            sensePredicted.append('product')
        else:
            senseActual.append('product')
            sensePredicted.append('phone')


answerList = pd.Series(sensePredicted, name = 'Prediction')
keyList = pd.Series(senseActual, name = 'Actual')
confusionMatrix = pd.crosstab(answerList, keyList, margins = True)

print("Overall accuracy of tagging: " + str(accuracy) + "%")
print("Overall accuracy of baseline sense: " + str(answerBaseline) + "%" + "\n")
print(confusionMatrix)







