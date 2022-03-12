
import sklearn
import numpy as np
import pandas as pd
import pickle
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

import preprocessor as p

def clean_tweets(df):
  tempArr = []
  for line in df:
    # send to tweet_processor
    tmpL = p.clean(line)
    # remove puctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)
  return tempArr

train_tweet = clean_tweets(train["tweet"])
train_tweet = pd.DataFrame(train_tweet)

train["clean_tweet"] = train_tweet

# clean the test data and append the cleaned tweets to the test data
test_tweet = clean_tweets(test["tweet"])
test_tweet = pd.DataFrame(test_tweet)
# append cleaned tweets to the training data
test["clean_tweet"] = test_tweet

# compare the cleaned and uncleaned tweets
test.tail()

from sklearn.model_selection import train_test_split

# extract the labels from the train data
y = train.label.values

# use 70% for the training and 30% for the test
x_train, x_test, y_train, y_test = train_test_split(train.clean_tweet.values, y, 
                                                    stratify=y, 
                                                    random_state=1, 
                                                    test_size=0.3, shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)

from sklearn import svm

# classify using support vector classifier
svm = svm.SVC(kernel = 'linear', probability=True)
# fit the SVC model based on the given training data/ training the model using fit method 
prob = svm.fit(x_train_vec, y_train)
# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(x_test_vec)

from sklearn.metrics import accuracy_score
print("Accuracy score for SVC is: ", accuracy_score(y_test, y_pred_svm) * 100, '%')



#Updated Code Here
#Saving SVM model 
filename = 'final_predict_model.sav'
print("Saving Modelss....")
pickle.dump(svm, open(filename, 'wb'))

#Saving Vectorization Model 

filename = 'vectorize_model.sav'
pickle.dump(vectorizer, open(filename, 'wb'))


