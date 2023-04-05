import pandas as pd
import numpy as np
import os
import time
from newsdataapi import NewsDataApiClient
import schedule
import datetime


def get_data():
    # API key authorization, Initialize the client with your API key
    # https://newsdata.io/api/1/news?apikey=YOUR_API_KEY
    # https://newsdata.io/api/1/archive?apikey=YOUR_API_KEY
    api = NewsDataApiClient(apikey="pub_16664f533f2ba832111bf1f9db80e7b44c889")
    def sleep_time(hour, min, sec):
        return hour * 3600 + min * 60 + sec
    second = sleep_time(0, 0, 2)
    try:
        j=0
        response = api.news_api(language="en",country='us')
        list=response['results']
        while j <150:
            pageurl=response['nextPage']
            response = response = api.news_api(language="en",country='us',page=pageurl)
            list.extend(response['results'])
            j=j+1
            time.sleep(second)
        df=pd.DataFrame(list)
    except:
        df=pd.DataFrame(list)
    
    list_cat=[]
    for i in df['category']:
        list_cat.append(i[0])
    df['cat_pro']=list_cat
    df.cat_pro.value_counts()
    df=df.fillna(' ')
    df['text']=df["title"] + " " + df["description"]+ " " + df["content"]
    df1=df[df['cat_pro'].notna()]
    df1=df[df['cat_pro']!='top']
    
    today=datetime.date.today()
    formatted_today=today.strftime('%y%m%d')
    
    categories = df1['cat_pro'].unique()
    ## Consolidate some of the groups to Mitigate class imbalance
    def groupper(dataset, grouplist, name):
        for ele in categories:
            if ele in grouplist:
                dataset.loc[dataset['cat_pro'] == ele, 'cat_pro'] = name
        
    groupper( dataset = df1, grouplist= ['science','technology'] , name =  'Sci and Tech')
    groupper( dataset = df1, grouplist= ['sports'] , name =  'Sport')
    groupper( dataset = df1, grouplist= ['entertainment'] , name =  'Entertainment')
    groupper( dataset = df1, grouplist= ['business'] , name =  'Business')
    groupper( dataset = df1, grouplist= ['politics'] , name =  'Politics')
    groupper( dataset = df1, grouplist= ['environment','world','health','food'] , name =  'Daily life news')
    df.to_csv('./Datacollect/test/'+'testnews'+formatted_today+'.csv')
    df1.to_csv('./Datacollect/train/'+'trainnews'+formatted_today+'.csv')
    import tensorflow as tf
    new_model = tf.keras.models.load_model('./saved_model/my_model_1', compile=False)
    from tensorflow.keras import layers
    from tensorflow import keras
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    loss = 'categorical_crossentropy'
    optim = keras.optimizers.Adam(learning_rate=0.0003)
    metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    new_model.compile(loss=loss, optimizer=optim, metrics=metrics)
    df_t= df
    import pickle
    # loading
    with open('./saved_model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    test_sentences = df_t.text.to_numpy()
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences, maxlen=500, padding="post", truncating="post")
    pred = new_model.predict(test_padded)
    list_label = []
    list1 = ['Business', 'Daily life news', 'Entertainment', 'Politics', 'Sci and Tech', 'Sport']
    for i in pred:
        list_label.append(list1[np.argmax(i)])
    df_t['cat_pro'] = list_label
    df_t.to_csv('./Datacollect/cat/'+'catnews'+formatted_today+'.csv')

schedule.every().day.at("14:40").do(get_data)
while True:
    schedule.run_pending()
    time.sleep(1)






