#from crypt import methods
import email
from flask import Flask, render_template,request, jsonify
from gevent import config
import numpy as np
import pickle
import re
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer
import pyrebase

firebaseConfig = {
  'apiKey': "AIzaSyAvtAddMEXIzvl4VwK9DTbLxVwEEuyF4Uo",
  'authDomain': "hatespeechdetection-abba8.firebaseapp.com",
  'projectId': "hatespeechdetection-abba8",
  'storageBucket': "hatespeechdetection-abba8.appspot.com",
  'messagingSenderId': "125967486520",
  'appId': "1:125967486520:web:87e3d104b5b9b6837720cd",
  'databaseURL': ""
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

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



#landing page
@app.route("/")
@app.route("/landingpg")
def landing():
    return render_template('landinpg.html')

# app = Flask(static_folder='C:\\FYP\\Hate-Speech-Detection\\assets')


#index page
@app.route("/index", methods= ["GET","POST"])
def hello():
    if request.method == 'POST':
        email = request.form['user_email']
        password = request.form['user_pwd']
        try:
            auth.sign_in_with_email_and_password(email,password)
            user_info = auth.sign_in_with_email_and_password(email,password)
            account_info = auth.get_account_info(user_info['idToken'])
           # if account_info['user'][0]['emailVerified'] ==False:
              #  verify_message = 'Please verify your email'
              #  return render_template('index.html', umessage= verify_message)
            return render_template('home.html')
        except:
            unsuccessful = 'Please check your credentials'
            return render_template('index.html', umessage= unsuccessful)
    return render_template("index.html")

#home page
@app.route("/home")
def home():
    return render_template('home.html')
#new user account create
@app.route("/create_account", methods =['GET','POST'])
def create_account():
    if request.method == 'POST':
        pwd0 = request.form['user_pwd0']
        pwd1 = request.form['user_pwd1']
        if pwd0 == pwd1:
            try:
                email = request.form['user_email']
                password = request.form['user_pwd1']
                new_user = auth.create_user_with_email_and_password(email, password)
                auth.send_email_verification(new_user['idToken'])
                return render_template("verify_email.html")
            except:
                existing_account = 'You have existing account under this email '
                return render_template("create_account.html", exist_message = existing_account)

    return render_template("create_account.html")

#reset password
@app.route("/reset_password", methods =['GET','POST'])
def forget_password():
    if request.method == 'POST':
        email = request.form['user_email']
        auth.send_password_reset_email(email)
        return render_template('index.html')
    return render_template('reset_password.html')

#submit to analyse
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
            for status in tweepy.Cursor(api.user_timeline,  tweet_mode='extended',screen_name = username).items(10):
                text = status.full_text
                if hasattr(status, "retweeted_status"):
                    text = status.retweeted_status.full_text                    
                tempArr = CleanText(text)
                corpus = [str(tempArr)]
                vectors = vec_loaded.transform(corpus)
                y = model.predict(vectors)
                if y == 0:
                    result = 'No Hate'
                else:
                    result = 'Hate'
                    
                result_dict = { 
                    'text':text,
                    'result':result,
                    'created_at': status.created_at.replace(tzinfo=None)
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

#function to clean raw input text
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
