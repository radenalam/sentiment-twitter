import csv
from pathlib import Path

import re
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import tweepy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

st.set_page_config(
    page_title="Sentiment Analisis Bahasa Indonesia",
    page_icon="🧠")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1588421357574-87938a86fa28?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 




def sentiment_indonesia(text):
    text_twitter = TextBlob(text)
    translate_tweet = text_twitter.translate(from_lang='id', to='en')
    sentiment = translate_tweet.sentiment

    if sentiment.polarity > 0:
        return 'Positive 😁'
    elif sentiment.polarity < 0:
        return 'Negative 🙁'
    else:
        return'Netrall 😐'

def sentiment_english(text):
    text_twitter = TextBlob(text)
    sentiment = text_twitter.sentiment

    if sentiment.polarity > 0:
        return 'Positive 😁'
    elif sentiment.polarity < 0:
        return 'Negative 🙁'
    else:
        return'Netrall 😐'





def main():
    st.title("Sentiment Analysis NLP App")
    st.write(
        "Sentiment analysis adalah proses penggunaan text analytics untuk mendapatkan berbagai sumber data dari internet dan beragam platform media sosial." 
        "Tujuannya adalah untuk memperoleh opini dari pengguna yang terdapat pada platform tersebut."

        )
    st.subheader("Bahasa Indonesia")

    menu = ["Home" , "About", "Twitter"]
    st.sidebar.title("Sentiment App")
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == "Home":
        with st.form ("nlpForm"):
            raw_text = st.text_area("Masukan Kalimat disini")
            submit_button = st.form_submit_button (label='Analyze')

            #Layout
            col1,col2 = st.columns(2)
            if submit_button:
                with col1:
                    st.info('Result')

                    inputtext = TextBlob(raw_text)
                    hasiltranslate = inputtext.translate(from_lang='id', to='en')
                    sentiment = hasiltranslate.sentiment
                    
                    #emoji
                    if sentiment.polarity > 0:
                        st.markdown('Sentiment:: Positive 😁')
                    elif sentiment.polarity < 0:
                        st.markdown('Sentiment:: Negative 🙁')
                    else:
                        st.markdown('Sentiment:: Netrall 😐')

                    #dataframe
                    result_df = convert_to_df(sentiment)
                    st.dataframe(result_df)

                    #Visual
                    c = alt.Chart(result_df).mark_bar().encode(
                        x='metric',
                        y='value',
                        color='metric'
                    )
                    st.altair_chart(c,use_container_width=True)

                with col2:
                    st.info('Token Sentimen')

                    token_sentiments = analyze_token_sentiment(hasiltranslate)
                    st.write(token_sentiments)

    elif choice == "Twitter":
            st.header("Crawling Twitter")
        
            query = st.text_area("Masukan keyword untuk twitter")
            English = 'lang:en'
            Indonesia = 'lang:id'
            options = st.selectbox(
                'Pilih Bahasa:',
                ('indonesia', 'english'))


            jumlah = st.slider("Jumlah tweet yang diambil", min_value=0, max_value=150,value=20,)
            submit_button2 = st.button(label='Cari')
            if submit_button2:
                # Masukkan Twitter Token API
                client = tweepy.Client(bearer_token=config.BEARER_TOKEN)
                # Query pencarian
                hasil = client.search_recent_tweets(query=query+' '+options, max_results=jumlah, tweet_fields=['created_at','lang'], user_fields=["username"], expansions=["author_id"])
                text = []
                created_at = []
                username = []
                users = {u['id']: u for u in hasil.includes['users']}



                #menambahkan tweet ke array
                for tweet in hasil.data:
                    user = users[tweet.author_id]
                    text.append(tweet.text)
                    created_at.append(tweet.created_at)
                    username.append(user.username)


                dictTweets = {"text":text, "created_at":created_at, "username":username}
                df = pd.DataFrame(dictTweets,columns=["username","text","created_at"])
                st.title('Original :')
                df

                csv = convert_df(df)
                st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='large_df.csv',
                        mime='text/csv',
                    )
                

                st.title('Clean text :')
                df['text'] = df['text'].apply(cleanTxt) # Membersihkan text dari simbol2
                

                csv = convert_df(df)
                st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='large_df.csv',
                        mime='text/csv',
                    )
                
                if options == 'indonesia':
                    df['sentiment'] = df['text'].apply(sentiment_indonesia) #proses sentiment
                    
                else:
                    df['sentiment'] = df['text'].apply(sentiment_english) #proses sentimen
                
                df

    else:
        st.subheader('About')


def convert_to_df(sentiment):
    sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity} 
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i.translate(from_lang='en', to='id'))
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i.translate(from_lang='en', to='id'))
            neg_list.append(res)
        else:
            neu_list.append(i.translate(from_lang='en', to='id'))

    result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return result

def cleanTxt(text):
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub(':', '', text) # Removing ':' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub(r'http\S+', '', text) # Removing hyperlink
    text = text.lower() # mengecilkan huruf
    text = re.sub(r'\B@(?!(?:[a-z0-9.]*_){2})(?!(?:[a-z0-9_]*\.){2})[._a-z0-9]{3,24}\b', '', text) #Removing @mentions
          
    return text

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if __name__ == '__main__':
    main()