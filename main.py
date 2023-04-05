import streamlit as st
st.set_page_config(page_title='YourNews', page_icon='🖖')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

#python -m streamlit run main.py
import numpy as np
import time
import pandas as pd
import datetime
today=datetime.date.today()
formatted_today=today.strftime('%y%m%d')
yesterday = today - datetime.timedelta(days=1)
formatted_yesterday = yesterday.strftime('%y%m%d')

#df=pd.read_csv('./Datacollect/cat/catnews'+formatted_today+'.csv')
df1=pd.read_csv('./Datacollect/cat/catnews'+formatted_yesterday+'.csv')#正常用上边那个
try:
    df2=pd.read_csv('./Datacollect/cat/catnews'+formatted_today+'.csv')#正常用上边那个
    df=df1.append(df2)
except:
    df=df1
df.replace({"content":{' ':'Details are on the way!'},"description":{' ':'Details are on the way!'}},inplace=True)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ibb.co/kcnrtHx/image.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.sidebar.title('Choose your own Space!!')
add_selectbox = st.sidebar.radio(
    "Which topic are you interested? Choose one below~",
    ("Sport", "Entertainment", "Daily life news", "Sci and Tech", "Business", "Politics")
)

if add_selectbox == "Sport":
    try:
        df1=df[df['cat_pro']=='Sport']
        if st.button('Get today Sport news (click to refresh)'):
            df2 = df1.sample(n=1)
            st.title(df2['title'].to_list()[0])
            st.markdown(df2['creator'].to_list()[0])
            st.subheader('Description')
            st.write(df2['description'].to_list()[0])
            st.subheader('Content')
            st.write(df2['content'].to_list()[0])
    except:
        st.subheader('No such news today~')

if add_selectbox == "Entertainment":
    try:
        df1=df[df['cat_pro']=='Entertainment']
        if st.button('Get today Entertainment news (click to refresh)'):
            df2 = df1.sample(n=1)
            st.title(df2['title'].to_list()[0])
            st.markdown(df2['creator'].to_list()[0])
            st.subheader('Description')
            st.write(df2['description'].to_list()[0])
            st.subheader('Content')
            st.write(df2['content'].to_list()[0])
    except:
        st.subheader('No such news today~')

if add_selectbox == "Daily life news":
    try:
        df1=df[df['cat_pro']=='Daily life news']
        if st.button('Get today Daily life news (click to refresh)'):
            df2 = df1.sample(n=1)
            st.title(df2['title'].to_list()[0])
            st.markdown(df2['creator'].to_list()[0])
            st.subheader('Description')
            st.write(df2['description'].to_list()[0])
            st.subheader('Content')
            st.write(df2['content'].to_list()[0])
    except:
        st.subheader('No such news today~')

if add_selectbox == "Sci and Tech":
    try:
        df1=df[df['cat_pro']=='Sci and Tech']
        if st.button('Get today tech and science news (click to refresh)'):
            df2 = df1.sample(n=1)
            st.title(df2['title'].to_list()[0])
            st.markdown(df2['creator'].to_list()[0])
            st.subheader('Description')
            st.write(df2['description'].to_list()[0])
            st.subheader('Content')
            st.write(df2['content'].to_list()[0])
    except:
        st.subheader('No such news today~')

if add_selectbox == "Business":
    try:
        df1=df[df['cat_pro']=='Business']
        if st.button('Get today business news (click to refresh)'):
            df2 = df1.sample(n=1)
            st.title(df2['title'].to_list()[0])
            st.markdown(df2['creator'].to_list()[0])
            st.subheader('Description')
            st.write(df2['description'].to_list()[0])
            st.subheader('Content')
            st.write(df2['content'].to_list()[0])
    except:
        st.subheader('No such news today~')

if add_selectbox == "Politics":
    try:
        df1=df[df['cat_pro']=='Politics']
        if st.button('Get today Politics news (click to refresh)'):
            df2 = df1.sample(n=1)
            st.title(df2['title'].to_list()[0])
            st.markdown(df2['creator'].to_list()[0])
            st.subheader('Description')
            st.write(df2['description'].to_list()[0])
            st.subheader('Content')
            st.write(df2['content'].to_list()[0])
    except:
        st.subheader('No such news today~')