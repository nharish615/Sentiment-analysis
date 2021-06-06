# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import nltk
import csv
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import os 
import matplotlib.pyplot as plt
import pandas as pd 
import re

pos=0
neg=0
neu=0


def cleanText(text):
	text=re.sub(r'@[A-Za-z0-9]+','',text)#remove @ symbl
	text=re.sub(r'#','',text)#removing hash tag
	text=re.sub(r'RT[\s]+','',text)
	text=re.sub(r'https?:\/\/\S+','',text)
	text=re.sub(r'\\n','',text)
	text=re.sub(r':','',text)
	return text

def getSubjectivity(text):#create a function to get the subjectivity 
	return TextBlob(text).sentiment.subjectivity
			
def getPolarity(text):#create a function to get the  polarity
	return TextBlob(text).sentiment.polarity
	

def getAnalysis(score):#create a function to compute the nrgative ,neutral and positive
    if score < 0:
        global neg,pos,neu
        neg=neg+1
        return 'Negative'
    elif score==0:
        neu=neu+1
        return 'Neutral'
    else:
        pos=pos+1
        return 'Positive'


negative = []
with open("words_negative.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        negative.append(row)
        
positive = []
with open("words_positive.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        positive.append(row)
	
	
nltk.download('punkt')
def sentiment(text):
    temp = [] #
    text_sent = nltk.sent_tokenize(text)
    for sentence in text_sent:
        n_count = 0
        p_count = 0
        sent_words = nltk.word_tokenize(sentence)
        for word in sent_words:
            for item in positive:
                if(word == item[0]):
                    p_count +=1
            for item in negative:
                if(word == item[0]):
                    n_count +=1

        if(p_count > 0 and n_count == 0): #any number of only positives (+) [case 1]
            #print "+ : " + sentence
            temp.append(1)
        elif(n_count%2 > 0): #odd number of negatives (-) [case2]
            #print "- : " + sentence
            temp.append(-1)
        elif(n_count%2 ==0 and n_count > 0): #even number of negatives (+) [case3]
            #print "+ : " + sentence
            temp.append(1)
        else:
            #print "? : " + sentence
            temp.append(0)
    return temp

app = Flask(__name__)

@app.route('/')
def home():
    if os.path.exists("static/pie.png"):
        os.remove("static/pie.png")
    return render_template('index.html')
	
@app.route('/tweet')
def tweet_():
    if os.path.exists("static/pie.png"):
            os.remove("static/pie.png")
    return render_template('tweet.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if os.path.exists("static/pie.png"):
            os.remove("static/pie.png")
        message = request.form['message']
        data = cleanText(message)
        sente=sentiment(data)
        sent=0
        nn=0
        ne=0
        po=0
        for i in sente:
            if(i==0):
                nn+=1
            elif(i==-1):
                ne+=1
            else:
                po+=1
        if(ne > po and ne > nn):
            sent=-1
        elif(po > ne and po>nn):
            sent=1
        else:
            sent=0
        return render_template('result.html', prediction=sent)

@app.route('/tweet_result', methods=['POST'])
def tweet_result_():
    if request.method == 'POST':
        message = request.form['message']
        consumer_key='d9Ksoz6Wb1jD0mqbW8rjaSNb7'
        consumer_secret='pHXnVSJeLbOxaYlbOR7BWFdDNhZSF6IzegZV87qUSUqy6Qe8qG'
        access_token='3648603434-dGRu1nHet22tdoYeqaAGoN8MyZrNw9oXZQvGZUD'
        access_token_secret='PZ8pcQBCb5zVPLRQNVQZc3Yzi0rz1wPef6O7RO7gzcvOf'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
        auth.set_access_token(access_token, access_token_secret) 
        
        api = tweepy.API(auth, wait_on_rate_limit=True)
        #print(api)
        
        #extract 15 tweets from tweeter account
        posts=api.search(message ,count=10,long="english",tweet_mode="extended")
        
        #convert df
        df=pd.DataFrame([tweet.full_text for tweet in posts],columns=['Tweets'])
        
        #create a function to clean the tweets
        df['Tweets']=df['Tweets'].apply(cleanText)
			
		#create a two new columns
        df['Subjectivity']=df['Tweets'].apply(getSubjectivity)
        df['Polarity']=df['Tweets'].apply(getPolarity)
        
        
        df['Analysis']=df['Polarity'].apply(getAnalysis)
        df=df[['Tweets','Analysis']]
        df=df.values.tolist()
        df=df[:5]
        print(pos,neg,neu)
        plt.xlabel("Results")
        plt.ylabel("")
        plt.pie([pos,neg,neu],labels=['Positive','Negative','Neutral'],autopct="%1.1f%%")
        try:
            plt.savefig('static/pie.png')
        except:
            pass
        plt.close()
        #return render_template('tweet_result.html', tweets=df['Tweets'],results=df['Analysis'])
        return render_template('tweet_result.html', tweets=df,leng=len(df),topic=message)


if __name__ == '__main__':
	app.run(debug=True)
