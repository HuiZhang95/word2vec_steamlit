import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import pickle
from tensorflow.keras.models import load_model


st.title("model of Word2Vec")

with open('vocab_infor.pkl','rb') as f:  # Python 3: open(..., 'rb')
    word2idx, idx2word, vocab_size = pickle.load(f)


# embedding_dim = 300
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(vocab_size, activation='softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.load_weights("word2vec.h5")

model = load_model('word2vec.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Similarity is a metric which measures the distance between two words. This distance represents the way 
# words are related to each other
vectors = model.layers[0].trainable_weights[0].numpy()
import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])

# print_closest('zombie')

user_input_word = st.text_input("Input a word: ", 'love')
user_input_number = st.number_input("number of similar words: ", 1)

output_st = print_closest(user_input_word, user_input_number)

st.write('Here is the list of similar words :\n', output_st = 'romantic')