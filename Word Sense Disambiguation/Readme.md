# Programming Assignment 4: Word Sense Disambiguation

## **Summary**

This program detects whether a sentence uses the word "line" in the correct sense. The correct word sense will be either phone or product. A bag-of-words representation of the training data and other features were used as a model for our program to learn from. Using this model, the program reads an unannotated corpus and predicts whether an instance is more in context of the word "product" or "phone". The accuracy of the model can be seen within the description of wsd.py 

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

* To run wsd.py, enter the command : ```python wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt```

  * A list of predicted word sense will be outputted to my-line-answers.txt which you can view in your folder
  
    [![image.png](https://i.postimg.cc/kgwNynQJ/image.png)](https://postimg.cc/pmhnWMc4)

* To run scorer.py, enter the command : ```python scorerWSD.py my-line-answers.txt line-key.txt```
  
  * This will output wsd.py's model accuracy and a confusion matrix
  
    [![image.png](https://i.postimg.cc/gJ3nNDCV/image.png)](https://postimg.cc/YGqrS1q0)
  

