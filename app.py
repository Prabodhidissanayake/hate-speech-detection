#from flask import Flask, render_template, request
#import pickle
#import numpy as np

#model = pickle.load(open('hateSpeechModel.pickle', 'rb'))

#app = Flask(__name__)


#@app.route('/')
#def man():
   # return render_template('home.html')

#@app.route('/predict', methods=['POST'])
#def home():
 #text1 = request.form['tweet']

 #pred = model.predict(text1)
 #return render_template('after.html', data=pred)


#if __name__ =="__main__":
 #   app.run(debug=True)


from flask import Flask, render_template,request, jsonify
import numpy as np
import pickle
import re
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer
# import sklearn
# from sklearn import svm
# import pandas as pd
#from model import prob


app = Flask(__name__)
# import required libraries
import tweepy
import time
import datetime
import pandas as pd
api_key ="p9btWOq38TXczri4AfCSqJjkx"
# api secret key
api_secret_key = "6eVs0lKKfp6vfnNenY8wil8PmXlVSN7g1VNIVPawG8qXQf0dLf"
# access token
access_token = "1499855933974978570-apcsJqVHjqsxOhPl4wWTIhF3L9x9lc"
# access token secret
access_token_secret = "iILNydxaee0hl8WI8ZpOHL5MUqIWg3Or53nZnyZEMfGE2"
# authorize the API Key




@app.route("/")

def hello():
    return render_template("home.html")

@app.route("/submit", methods =['POST'])


def submit():
    filename_model = 'final_predict_model.sav'
    filename_vectorizer = 'vectorize_model.sav'
    model = clf2 = pickle.load(open(filename_model,'rb'))
    vec_loaded = pickle.load(open(filename_vectorizer,'rb'))
    if request.method == "POST":  
        text1 = request.form['tweet']
        Resultlist = []
        newlist = []
        inputtext = ''
        inputtype = request.form['IsText']
        
        if inputtype == 'Username':
            username = text1
            authentication = tweepy.OAuthHandler(api_key, api_secret_key)
            authentication.set_access_token(access_token, access_token_secret)
            api = tweepy.API(authentication, wait_on_rate_limit=True)
            for tweet in tweepy.Cursor(api.user_timeline, screen_name = username).items(10):
                text = tweet.text
                tempArr = CleanText(text)
                corpus = [str(tempArr)]
                vectors = vec_loaded.transform(corpus)
                y = model.predict(vectors)
                if y == 0:
                    result = 'No Hate'
                else:
                    result = 'Hate'
                result_dict = { 
                    'text':text[:30],
                    'result':result,
                    'created_at': tweet.created_at.replace(tzinfo=None)
                }
                Resultlist.append(result_dict)

            try:
                print(result)
            except:
                result = 'Username Not Found'
                inputtext = text1
        else:
            text = text1
            tempArr = CleanText(text)
            corpus = [str(tempArr)]
            vectors = vec_loaded.transform(corpus)
            y = model.predict(vectors)
            inputtext = text
            if y == 0:
                result = 'No Hate'
            else:
                result = 'Hate'
    
        return render_template("submit.html", resultList = Resultlist,result=result,inputtext=inputtext)


def CleanText(rawtext):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
    REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
    tempArr = []
    tmpL = p.clean(rawtext)
    # remove puctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)


    return tempArr

if __name__ == "__main__":
    app.run(debug=True) 
