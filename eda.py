import streamlit as st
# library untuk membaca dataset
import pandas as pd
# library untuk visualisasi
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import seaborn as sns
# library untuk import regex
import re
# library untuk download stopword
import nltk 
from nltk.corpus import stopwords
# library untuk tokenisasi kata
from nltk.tokenize import word_tokenize
from collections import Counter
# Define Stemming
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.stem import SnowballStemmer

def run():
    # Membuat title 
    st.title('Aplikasi Prediksi Sentimen Post (Tweet) Pengguna Twitter')

    # Membuat sub header
    st.subheader('Halaman ini berisi Exploratory Data Analysis (EDA) mengenai dataset Post (Tweet) Pengguna Twitter sampai dengan Tahun 2017')

    # Menampilkan teks
    st.write('Proyek ini dibuat untuk melakukan analisa sentimen pengguna berdasarkan dari post (tweet) dari pengguna sosial media di Twitter (X) pada tahun 2017 menggunakan model machine learning. Dataset diambil dari website Kaggle mengenai sentimen Tweet dalam bahasa Inggris.')
    st.write(' Model ini akan digunakan untuk membantu para moderator komunitas di Twitter untuk mencegah cyberbullying dan memberikan sanksi atau hukuman yang setimpal pada pengguna tersebut supaya komunitas sosial media lebih sehat dan nyaman untuk semua orang.')
    st.write('Model yang digunakan akan menggunakan algoritma Natural Language Processing dengan basis Artificial Neural Network dan metode evaluasinya akan dilihat dari nilai loss dan akurasi dari model')
    # Menampilkan gambar
    data = mpimage.imread('Twitter-Logo-2010.png')
    st.image(data, caption='EDA Tweet Pengguna Twitter')
    st.write('# Data Loading')
    df = pd.read_csv('Tweets.csv')
    st.write('Tampilan dataframe')
    st.dataframe(df)
    st.write('# Exploratory Data Analysis')
    st.subheader('1. Jumlah sentimen pada dataset')
    st.write('Jumlah setiap sentimen dalam dataset:')
    st.write(df["sentiment"].value_counts())
    st.write('Sentimen tweet umumnya bersifat netral. Selain itu, terdapat sentimen positif dan negatif yang hampir seimbang')
    st.write('Selanjutnya adalah visualisasi data jumlah sentimen')
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="sentiment"
    )
    plt.title("Distribusi sentimen")
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah Tweet")
    plt.show()
    st.pyplot(fig)
    st.write('Dapat dilihat dari distribusi diatas, bahwa sentimen negatif dan positif hampir seimbang. Namun untuk sentimen netral yang berjumlah paling banyak. Walaupun terdapat jumlah perbedaan sekitar 4000 tweet, secara keseluruhan data sudah cukup seimbang. Sehingga tidak perlu dilakukan penyeimbangan data.')
    st.write('')

    st.subheader('2. Melihat tampilan 10 tweet pertama dengan masing-masing jenis sentimen')
    st.write('Sentimen Netral')
    st.write(df[df['sentiment'] == 'neutral'].head(10))
    st.write('Sentimen Negatif')
    st.write(df[df['sentiment'] == 'negative'].head(10))
    st.write('Sentimen Positif')
    st.write(df[df['sentiment'] == 'positive'].head(10))
    st.write('Dari hasil diatas, dapat diketahui bahwa tweet dengan sentimen netral berisi tweet yang bersifat informatif. Tweet dengan sentimen negatif berisi tweet yang menunjukkan kesedihan atau kemarahan dari pengguna. Sendangkan tweet dengan sentimen positif menunjukkan rasa bahagia dan takjub dari penggunanya.')
    st.write('')

    st.subheader('3. Jumlah kata dalam setiap tweet')
    st.write(df["text"].dropna().str.len().describe())
    st.write('Rata-rata jumlah kata dalam setiap tweet adalah 68 kata, dengan jumlah tweet paling sedikit adalah 3 kata dalam sebuah tweet, dan jumlah terbanyak adalah 141 kata dalam 1 tweet. Nilai Median adalah 64 kata dalam 1 tweet.')
    st.write('')

    st.subheader('4. Jumlah kata dalam setiap tweet per setimen')
    st.write('Sentimen Netral')
    st.write(df.loc[df['sentiment'] == 'neutral', 'text'].dropna().str.len().describe())
    st.write('Sentimen Negatif')
    st.write(df.loc[df['sentiment'] == 'negative', 'text'].dropna().str.len().describe())
    st.write('Sentimen Positif')
    st.write(df.loc[df['sentiment'] == 'positive', 'text'].dropna().str.len().describe())
    st.write('Dari hasil diatas, diketahui bahwa:')
    st.write('- Rata-rata jumlah kata paling sedikit adalah bila sentimen bersifat netral. Sentimen positif dan negatif menunjukkan rata-rata jumlah kata hampir sama di 70 kata')
    st.write('- Jumlah kata paling sedikit untuk sentimen netral adalah 3 kata. Sedangkan untuk sentimen positif dan negatif paling sedikit terdapat 5 kata')
    st.write('- Jumlah kata paling banyak untuk sentimen positif adalah 138 kata. Sedangkan untuk sentimen negatif dan netral adalah 141 kata.')
    st.write('- Median jumlah kata paling kecil adalah untuk sentimen netral di 59 kata. Sedangkan untuk sentimen negatif adalah 66 kata dan untuk sentimen positif adalah 67 kata')
    st.write('')

    st.subheader('5. 20 Kata yang paling sering muncul')
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
    eda_text = df["text"].dropna().apply(basic_clean_text)

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
    # tampilkan 20 kata dengan jumlah paling banyak
    st.write(word_frequency.most_common(20))
    st.write('Dari 20 kata terbanyak yang muncul di dalam tweet, kata "im" paling banyak muncul dengan jumlah 3024 kali')
    st.write('')
    
    st.subheader('6. Kata yang sering muncul untuk setiap sentimen')
    st.write('Sentimen Netral')
    neutral_text = df.loc[df['sentiment'] == 'neutral', 'text'].dropna().apply(basic_clean_text)
    neutral_words = []
    for text in neutral_text:
        words = text.split()

        words = [
            word
            for word in words
            if word not in stop_words
        ]

        neutral_words.extend(words)
    # variabel untuk menyimpan kata-kata yang sudah ditokenisasi beserta dengan jumlah berapa banyak kata tersebut muncul
    neutral_word_frequency = Counter(neutral_words)
    st.write(neutral_word_frequency.most_common(1))
    st.write('Sentimen Negatif')
    negative_text = df.loc[df['sentiment'] == 'negative', 'text'].dropna().apply(basic_clean_text)
    negative_words = []
    for text in negative_text:
        words = text.split()

        words = [
            word
            for word in words
            if word not in stop_words
        ]

        negative_words.extend(words)
    # variabel untuk menyimpan kata-kata yang sudah ditokenisasi beserta dengan jumlah berapa banyak kata tersebut muncul
    negative_word_frequency = Counter(negative_words)
    st.write(negative_word_frequency.most_common(1))
    st.write('Sentimen Positif')
    positive_text = df.loc[df['sentiment'] == 'positive', 'text'].dropna().apply(basic_clean_text)
    positive_words = []
    for text in positive_text:
        words = text.split()

        words = [
            word
            for word in words
            if word not in stop_words
        ]

        positive_words.extend(words)
    # variabel untuk menyimpan kata-kata yang sudah ditokenisasi beserta dengan jumlah berapa banyak kata tersebut muncul
    positive_word_frequency = Counter(positive_words)
    st.write(positive_word_frequency.most_common(1))
    st.write('Untuk sentimen netral dan negatif, kata yang paling sering muncul adalah kata "im", sedangkan untuk sentimen positif, kata yang paling sering muncul adalah "day".')
    st.write('')


if __name__ == '__main__':
    run()