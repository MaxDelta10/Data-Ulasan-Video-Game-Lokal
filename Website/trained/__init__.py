from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import joblib
import os.path

pwd = os.path.abspath(os.path.dirname(__file__))
sentimen = pd.read_csv(os.path.join(
    pwd, "Steamreviewpreprocessing.csv"), sep=";")


def train():
    global sentimen
    sentimen.head()
    sentimen = sentimen.astype({'Review': 'category', 'Review_Text': 'string'})
    sentimen.dtypes
    sentimen['Review'].value_counts()

    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(sentimen['Review_Text'].astype('U'))
    x_train, _, y_train, _ = train_test_split(
        text_tf, sentimen['Review'], test_size=0.5, random_state=0)

    clf = MultinomialNB().fit(x_train, y_train)
    clf.fit(x_train, y_train)

    joblib.dump(clf, os.path.join(pwd, 'trained_model.pkl'))


def predict(input_text):
    global sentimen
    clf = joblib.load(os.path.join(pwd, 'trained_model.pkl'))

    # Proses TF.IDF
    tf = TfidfVectorizer()
    tf.fit_transform(sentimen['Review_Text'].astype('U'))
    tf_input_text = tf.transform([input_text])

    # Predict the input text
    predicted_label = clf.predict(tf_input_text)
    return predicted_label
