import streamlit as st
# hilangkan semua warning dari kode
import warnings
warnings.filterwarnings('ignore')
# library untuk melakukan perhitungan matematika pada dataframe
import numpy as np

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
        text = st.text_input('Tweet Pengguna', value='-tweet pengguna-')

        submitted = st.form_submit_button('Predict')
    if submitted:
        data_inf = tf.constant([text], dtype=tf.string)

        predict = model(data_inf, training=False)
        predict = predict.numpy()
        label_mapping = {
            0: 'neutral',
            1: 'negative',
            2: 'positive'}

        predictions = predict
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_sentiments = [label_mapping[label] for label in predicted_labels]
        predicted_sentiments

        st.write('## Prediksi sentimen tweet: ',str(predicted_sentiments[0]))

       
if __name__ == '__main__':
  run()