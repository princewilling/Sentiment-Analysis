import pickle
import os
import numpy as np
import sys 
import streamlit as st
import sqlite3
from update import update_model

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# import HashingVectorizer from local dir
from vectorizer import vect



######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'trained_model.sav'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def classify_(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y]

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
        " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()



def main():

    ##giving a title
    st.title("Movie Review Classifier")

    ##getting input data from user
    review = st.text_area("PLEASE TYPE IN YOUR COMMENT OR REVIEW ON THIS MOVIE")

    ##code for prediction
    result, probability = '', ''

    ##creating a result button
    if st.button("Result of Review", ):
        result, probability = classify(review)

    if result == 'positive':
        text = "YOU REVIEW HAS BEEN SUBMITTED, IT IS A POSITIVE+VE REVIEW..."
        st.write(text)
        value = str(probability) + str(" %")
        st.write(value)
    elif result == 'negative':
        text = "YOU REVIEW HAS BEEN SUBMITTED, IT IS A NEGATIVE-VE REVIEW..."
        st.write(text)
        value = str(probability) + str(" %")
        st.write(value)
   
    
    
    ## get user feedback and update classifier
    inv_label = {'negative': 0, 'positive': 1}
    st.text("CLICK BUTTON BELOW IF REVIEW RESULT IS WRONG, OTHERWISE IGNORE")
    if st.button("INVALID"):
        review = review
        y = inv_label[classify_(review)]
        y = int(not(y))
        train(review, y)
        sqlite_entry(db, review, y)
        st.success("THANK YOU FOR YOUR FEED BACk")
    


if __name__ == '__main__':
    clf = update_model(db_path=db, model=clf,batch_size=10000)
    main()

