

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

ticker='^BSESN'
end=dt.datetime.today()
start=dt.datetime.today()-dt.timedelta(3650)

ohclv_data=pd.DataFrame()
ohclv_data=yf.download(ticker,start,end)
ohclv_data=ohclv_data.dropna()

x=ohclv_data.iloc[0:len(ohclv_data)-1,[0,1,2]]
y=np.asarray(ohclv_data.iloc[1:len(ohclv_data),4])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
z=regressor.predict((np.asarray(x.iloc[-1,[0,1,2]]).reshape(1,3)))
print("closing price is")
print(z)
print("with accuracy")
print(regressor.score(x_test,y_test))

x=ohclv_data.iloc[0:len(ohclv_data)-1,[1,2,4]]
y=np.asarray(ohclv_data.iloc[1:len(ohclv_data),0])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
z=regressor.predict((np.asarray(x.iloc[-1,[0,1,2]]).reshape(1,3)))
print("opening price is")
print(z)
print("with accuracy")
print(regressor.score(x_test,y_test))


df=pd.read_csv("india-news-headlines.csv")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
def comp_score(text):
   return analyser.polarity_scores(text)["compound"]   
df["sentiment"] = df["headline_text"].apply(comp_score)
a=df.groupby("publish_date").count()
a=np.asarray(a.iloc[:,0].values)
a=np.insert(a,0,0)
a=a.cumsum()
b=0
sentiment_score=[]
for b in range(0,len(a)-1):
    z=np.mean(df.iloc[a[b]:a[b+1],3])
    sentiment_score.append(z)
    
sentiment_score=sentiment_score[-len(x):]
d=ohclv_data.iloc[-len(sentiment_score):,:]
e=0
f=d.iloc[:,0].tolist()
g=d.iloc[:,4].tolist()
h=np.zeros(len(d))
for e in range(0,len(d)):
    if g[e]>=f[e]:
        h[e]=1
    else:
        h[e]=0
d["movement"]=h.reshape(len(h),1)
sentiment_score=np.array(sentiment_score)
sentiment_score=sentiment_score.reshape(len(sentiment_score),1)
X=sentiment_score
y=np.array(d["movement"]).reshape(len(d),1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from bs4 import BeautifulSoup
import requests

temp_dir=[]
url = "https://in.finance.yahoo.com/"
page = requests.get(url)
soup = BeautifulSoup(page.content,'html.parser')
table = soup.find_all("div", {"class" : "Py(14px) Pos(r)"})
for t in table:
    rows = t.find_all("div", {"class" : "Cf"})
    for row in rows:
      temp_dir.append(row.get_text())
        
        
temp_dir=pd.DataFrame(temp_dir)
temp_dir["sentiment"] = temp_dir.apply(comp_score)
x=np.mean(temp_dir["sentiment"])

print("Sentiment Score is")
print(x)
x=np.asarray(x).reshape(1,1)
if classifier.predict(x)>=0:
    print("Market should be bullish")
else: 
    print("Market should be bearish")





