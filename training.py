# =====================================================
# IMPORT LIBRARIES
# =====================================================

import json
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    Bidirectional, GlobalMaxPooling1D,
    GlobalAveragePooling1D, Concatenate,
    SpatialDropout1D
)

from tensorflow.keras.models import Model, load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns


# =====================================================
# LOAD DATASET
# =====================================================

with open("data.json","r",encoding="utf-8") as f:
    data=json.load(f)

df=pd.DataFrame(data)

print("Original dataset:",df.shape)


# =====================================================
# CLEAN DATASET
# =====================================================

df=df.dropna(subset=["input","emotion","response"])

df["emotion"]=df["emotion"].astype(str).str.strip()

df=df[df["emotion"]!=""]

def clean_text(text):

    text=str(text).lower()
    text=re.sub(r"[^a-zA-Z\s]","",text)
    text=re.sub(r"\s+"," ",text)

    return text.strip()

df["clean_text"]=df["input"].apply(clean_text)

# remove duplicates
df=df.drop_duplicates(subset="clean_text")

# remove conflicting labels
conflicts=df.groupby("clean_text")["emotion"].nunique()
conflicts=conflicts[conflicts>1]

df=df[~df["clean_text"].isin(conflicts.index)]

df=df.reset_index(drop=True)

print("Dataset after cleaning:",df.shape)


# =====================================================
# LABEL ENCODING
# =====================================================

label_encoder=LabelEncoder()

df["label"]=label_encoder.fit_transform(df["emotion"])

num_classes=len(label_encoder.classes_)

print("Emotions:",label_encoder.classes_)


# =====================================================
# TOKENIZATION
# =====================================================

max_words=10000
max_len=20

tokenizer=Tokenizer(num_words=max_words,oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])

sequences=tokenizer.texts_to_sequences(df["clean_text"])

X=pad_sequences(sequences,maxlen=max_len)
y=df["label"].values


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =====================================================
# MODEL ARCHITECTURE
# =====================================================

input_layer=Input(shape=(max_len,))

embedding=Embedding(max_words,128)(input_layer)

embedding=SpatialDropout1D(0.3)(embedding)

x=Bidirectional(LSTM(64,return_sequences=True))(embedding)

max_pool=GlobalMaxPooling1D()(x)
avg_pool=GlobalAveragePooling1D()(x)

x=Concatenate()([max_pool,avg_pool])

x=Dense(128,activation="relu")(x)

x=Dropout(0.5)(x)

output=Dense(num_classes,activation="softmax")(x)

model=Model(inputs=input_layer,outputs=output)

optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =====================================================
# TRAIN MODEL
# =====================================================

history=model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=25,
    batch_size=32
)


# =====================================================
# TRAINING GRAPHS
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train","Validation"])

plt.show()


plt.figure(figsize=(10,5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train","Validation"])

plt.show()


# =====================================================
# MODEL EVALUATION
# =====================================================

pred=model.predict(X_test)

y_pred=np.argmax(pred,axis=1)

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


# =====================================================
# PRECISION RECALL F1 SCORE GRAPH
# =====================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

# Remove accuracy / avg rows
report_df = report_df.iloc[:-3]

plt.figure(figsize=(10,6))

x = np.arange(len(report_df.index))

width = 0.25

plt.bar(x - width, report_df["precision"], width, label="Precision")
plt.bar(x, report_df["recall"], width, label="Recall")
plt.bar(x + width, report_df["f1-score"], width, label="F1 Score")

plt.xticks(x, report_df.index, rotation=45)

plt.ylabel("Score")
plt.xlabel("Emotion Classes")
plt.title("Precision, Recall and F1 Score per Emotion")

plt.legend()

plt.tight_layout()

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

model.save("emotion_model.keras")

with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

with open("label_encoder.pkl","wb") as f:
    pickle.dump(label_encoder,f)

print("Model saved successfully")

# =====================================================
# CLEAR DATASET FROM MEMORY
# =====================================================

responses=df[["clean_text","emotion","response"]].copy()

vectorizer=TfidfVectorizer()

tfidf_matrix=vectorizer.fit_transform(responses["clean_text"])

del df


# =====================================================
# LOAD MODEL FOR INFERENCE
# =====================================================

model=load_model("emotion_model.keras")

with open("tokenizer.pkl","rb") as f:
    tokenizer=pickle.load(f)

with open("label_encoder.pkl","rb") as f:
    label_encoder=pickle.load(f)


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

print("\nChatbot ready (type 'exit' to stop)\n")

while True:

    user=input("You: ")

    if user.lower()=="exit":

        print("Chatbot: Take care. I'm here whenever you need.")

        break

    emotion=predict_emotion(user)

    response=get_best_response(user,emotion)

    print("\nDetected Emotion:",emotion)

    print("Chatbot:",response)