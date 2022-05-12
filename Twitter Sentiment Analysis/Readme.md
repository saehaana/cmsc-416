# Programming Assignment 5: Twitter Sentiment Analysis

**Summary**

This program performs positive/negative sentiment analysis over tweets downloaded from twitter. A bag-of-words representation of the training data and other features were used as a model for our program to learn from. Using this model, the program reads an unannotated corpus and predicts whether a tweet is more positive or negative in sentiment. The accuracy of the model can be seen within the description of sentiment.py 

**Installation**

* Download all other files within this folder

* This program also requires you have [nltk](https://www.nltk.org/data.html) and nltk stopwords to run 

  **How to Install nltk**
  
  * Open command prompt
  
  * For nltk : type ```pip install nltk```

  * For nltk stopwords : type ```python -m nltk.downloader all```



**How To Use sentiment.py and scorer.py**

* Open command prompt and change directory to where you downloaded your files

* Enter the command : python sentiment.py sentiment-train.txt sentiment-test.txt my-model.txt > my-sentiment-answers.txt

* To view list of predicted sentiments open my-sentiment-answers.txt 

* 