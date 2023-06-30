import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')

def load_and_preprocess_data(filename):
    # Lade den Datensatz
    df = pd.read_csv(filename, delimiter='\t', encoding='utf-8')



    # Prüfe, ob es fehlende Werte gibt
    if df.isnull().values.any():
        print("Es gibt fehlende Werte im Datensatz. Diese werden entfernt.")
        df = df.dropna(subset=['Text', 'Source'])

    # Prüfe, ob es nicht-alphabetische Zeichen im Text gibt
    if df['Text'].str.contains('[^a-z\s]').any():
        print("Es gibt nicht-alphabetische Zeichen im Text. Diese werden entfernt.")
        # Setze alle Texte in Kleinbuchstaben um
        df['Text'] = df['Text'].str.lower()
        # Entferne alle nicht-alphabetischen Zeichen
        df['Text'] = df['Text'].str.replace('[^a-z\s]', '')

    # Entferne irrelevante Merkmale
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # Entfernen aller nicht Englische oder Deutsche Texte
    df = df[df['Language'].isin(['English', 'German'])]

    unique_sources = df['Source'].unique()
    print(unique_sources)

    # Entfernen von Stoppwörtern und Durchführen von Stemming
    stop_words = set(stopwords.words(['german', 'english']))
    stemmer = SnowballStemmer('english') # Ersetzen Sie 'english' durch 'german' für deutsche Texte
    df['Text'] = df['Text'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split() if word not in stop_words))

    # Erstellen Zielspalte für Populismus basierend auf der Quelle des Artikels
    df['Populism'] = df['Source'].apply(lambda x: 1 if x == 'Bild' or x == 'nypostonline.com' else 0)


    # Speichere den bereinigten Datensatz
    df.to_csv('data/processed/processed.csv', index=False)

    return df

# Verwendung
load_and_preprocess_data('data/raw/old-newspaper.tsv')
