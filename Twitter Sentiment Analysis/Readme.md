# Programming Assignment 5: Twitter Sentiment Analysis

## **Summary**

This program performs positive/negative sentiment analysis over tweets downloaded from twitter. A bag-of-words representation of the training data and other features were used as a model for our program to learn from. Using this model, the program reads an unannotated corpus and predicts whether a tweet is more positive or negative in sentiment. The accuracy of the model can be seen within the description of sentiment.py 

## **Installation**

* Download all other files within this folder

* This program also requires you have [nltk stopwords](https://www.nltk.org/data.html) to run 

  ### **How to Install nltk**
  
  * Open command prompt
  
  * For nltk : type ```pip install nltk```

  * For just nltk stopwords : type ```python -m nltk.downloader stopwords```

## **How To Use**

* Open command prompt and change directory to where you downloaded your files
  * e.g. If you downloaded the files to your Downloads folder then enter 'cd downloads'
  
    [![image.png](https://i.postimg.cc/P5n7qP3b/image.png)](https://postimg.cc/yknjQYzW)

* To run sentiment.py, enter the command : ```python sentiment.py sentiment-train.txt sentiment-test.txt my-model.txt > my-sentiment-answers.txt```

  * A list of predicted sentiments will be outputted to my-sentiment-answers.txt
  
    [![image.png](https://i.postimg.cc/wM7B3WBr/image.png)](https://postimg.cc/2LNYK7hd)

* To run scorer.py, enter the command : ```python scorer.py my-sentiment-answers.txt sentiment-test-key.txt ```
  
  * This will output sentiment.py's model accuracy and a confusion matrix
  
    [![image.png](https://i.postimg.cc/T1CrvN4H/image.png)](https://postimg.cc/JHHybxmb)
  
