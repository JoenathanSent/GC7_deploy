import streamlit as st
# hilangkan semua warning dari kode
import warnings
warnings.filterwarnings('ignore')
# library untuk membuat dan membaca dataframe
import pandas as pd
# library untuk melakukan perhitungan matematika pada dataframe
import numpy as np

# library untuk import regex
import re
# library untuk download stopword
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
# library untuk tokenisasi kata
from nltk.tokenize import word_tokenize
# Define Stemming
from nltk.stem import SnowballStemmer
from collections import Counter

# library untuk melakukan proses machine learning dengan metode Natural Language Processing
import tensorflow as tf
# library untuk melakukan encoding target sentimen
from tensorflow.keras.utils import to_categorical
# library untuk import pre-trained model NNLM
import tensorflow_hub as tf_hub
# library untuk import model yang sudah disimpan
from tensorflow.keras.models import load_model

url = 'https://www.kaggle.com/models/google/nnlm/TensorFlow2/id-dim128-with-normalization/1'
nnlm = tf_hub.KerasLayer(url, input_shape=[], dtype=tf.string, trainable=False)
import keras.src.utils.python_utils as keras_python_utils
keras_python_utils.nnlm = nnlm
model = load_model('model_nnlm.keras', safe_mode=False)

def run():
    with st.form(key='tweet_sentiment_2017_rmt_057'):
        tweet = st.text_input('Tweet Pengguna', value='-tweet pengguna-')

        data_inf = {
            'tweet': tweet 
        }
        stemmer = SnowballStemmer("english")
        # tokenisasi
        def basic_clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"#", "", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            # Tokenization
            tokens = word_tokenize(text)
    
            # Stopwords removal
            stpwds_en = list(set(stopwords.words('english')))
            stpwds_en.append('oh')
            stpwds_en.append('ohh')
            tokens = [word for word in tokens if word not in stpwds_en]
    
            # Stemming
            tokens = [stemmer.stem(word) for word in tokens]
    
            # Combining Tokens
            text = ' '.join(tokens)
            return text
    
        # drop baris dengan data kosong dan gunakan fungsi basic_clean_text pada setiap baris data
        eda_text = data_inf.apply(basic_clean_text)
    
        # list kosong untuk menyimpan kata yang sudah di tokenisasi
        all_words = []
        # variabel berisi stop word
        stop_words = set(stopwords.words("english"))
        # perulangan untuk menghapus kata yang merupakan stop word
        for text in eda_text:
            words = text.split()
    
            words = [
                word
                for word in words
                if word not in stop_words
            ]
    
            all_words.extend(words)
        # variabel untuk menyimpan kata-kata yang sudah ditokenisasi beserta dengan jumlah berapa banyak kata tersebut muncul
        word_frequency = Counter(all_words)
        submitted = st.form_submit_button('Predict')
    if submitted:
        data_inf = pd.DataFrame(data_inf, columns=['text'])
        data_inf['text_processed'] = data_inf['text'].apply(basic_clean_text)
        data_inf = np.array(data_inf['text_processed'])
        predict = model.predict(data_inf)
        label_mapping = {
            0: 'neutral',
            1: 'negative',
            2: 'positive'}

        predictions = predict
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_sentiments = [label_mapping[label] for label in predicted_labels]
        predicted_sentiments

        st.write('## Prediksi sentimen tweet: ',str(predicted_sentiments))

       
if __name__ == '__main__':
  run()