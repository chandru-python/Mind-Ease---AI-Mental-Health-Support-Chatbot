# =====================================================
# IMPORT LIBRARIES
# =====================================================

import json
import re
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# =====================================================
# LOAD DATASET
# =====================================================

with open("data.json","r",encoding="utf-8") as f:
    data=json.load(f)

df=pd.DataFrame(data)


# =====================================================
# CLEAN TEXT FUNCTION
# =====================================================

def clean_text(text):

    text=str(text).lower()

    text=re.sub(r"[^a-zA-Z\s]","",text)

    text=re.sub(r"\s+"," ",text)

    return text.strip()


df["clean_text"]=df["input"].apply(clean_text)


# =====================================================
# LOAD MODEL + TOKENIZER
# =====================================================

model=load_model("emotion_model.keras")

with open("tokenizer.pkl","rb") as f:
    tokenizer=pickle.load(f)

with open("label_encoder.pkl","rb") as f:
    label_encoder=pickle.load(f)


max_len=20


# =====================================================
# RESPONSE DATASET
# =====================================================

responses=df[["clean_text","emotion","response"]].copy()

vectorizer=TfidfVectorizer()

tfidf_matrix=vectorizer.fit_transform(responses["clean_text"])


# =====================================================
# CRISIS DATA
# =====================================================

crisis_response=data[-1]["crisisResponse"]

crisis_inputs=[clean_text(x["input"]) for x in data[-1]["crisisInputs"]]

crisis_vectorizer=TfidfVectorizer()

crisis_matrix=crisis_vectorizer.fit_transform(crisis_inputs)


# =====================================================
# EMOTION PREDICTION
# =====================================================

def predict_emotion(text):

    text=clean_text(text)

    seq=tokenizer.texts_to_sequences([text])

    seq=pad_sequences(seq,maxlen=max_len)

    pred=model.predict(seq)

    emotion=label_encoder.inverse_transform([np.argmax(pred)])

    return emotion[0]


# =====================================================
# CRISIS DETECTION
# =====================================================

def detect_crisis(text):

    text=clean_text(text)

    vec=crisis_vectorizer.transform([text])

    similarity=cosine_similarity(vec,crisis_matrix).max()

    if similarity>0.5:

        return True

    return False


# =====================================================
# RESPONSE RETRIEVAL
# =====================================================

def get_best_response(user_input,emotion):

    user_input=clean_text(user_input)

    user_vec=vectorizer.transform([user_input])

    cosine_sim=cosine_similarity(user_vec,tfidf_matrix).flatten()

    emotion_indices=responses[responses["emotion"]==emotion].index

    best_index=emotion_indices[np.argmax(cosine_sim[emotion_indices])]

    return responses.loc[best_index,"response"]


# =====================================================
# CHATBOT LOOP
# =====================================================

print("\nMental Health Chatbot Ready\n")

while True:

    user=input("You: ")

    if user.lower()=="exit":

        print("Chatbot: Take care of yourself.")

        break


    # CRISIS CHECK
    if detect_crisis(user):

        print("\nEmotion: crisis")

        print("Chatbot:",crisis_response)

        continue


    # NORMAL EMOTION
    emotion=predict_emotion(user)

    response=get_best_response(user,emotion)

    print("\nEmotion:",emotion)

    print("Chatbot:",response)