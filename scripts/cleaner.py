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

    # Entfernen aller nicht Englische, Deutsche oder Französische Texte
    df = df[df['Language'].isin(['English', 'German', 'French'])]

    unique_sources = df['Source'].unique()
    print(unique_sources)

    # Entfernen von Stoppwörtern und Durchführen von Stemming
    stop_words = set(stopwords.words(['german', 'english', 'french']))
    stemmer = SnowballStemmer('english') # Ersetzen Sie 'english' durch 'german' oder 'french' für deutsche oder französische Texte
    df['Text'] = df['Text'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split() if word not in stop_words))

    # Gewichtete Liste von populistischen Begriffen pro Sprache
    populist_terms = {
        'English': {'elite': 1.5, 'mainstream media': 2, 'people': 1, 'corrupt': 1.5, 'brothers sisters': 1, 'islamic world': 1, 'dear friends': 1, 'religious popularity': 1, 'past years': 1, 'islamic state': 1, 'islamic revolution': 2, 'years ago': 1, 'come back': 1, 'would like': 1, 'religious popularity': 2},
        'German': {'elite': 1.5, 'lügenpresse': 2, 'volk': 2, 'korrupt': 1.5, 'brüder schwestern': 1.5, 'islamische welt': 1, 'liebe freunde': 1, 'vergangene jahre': 1, 'islamische revolution': 1.5, 'verrat bevölkerung': 2, 'verrat volk':2,'wahrheit unterdrückt': 2, 'regierung versagt': 2, 'frau präsidentin': 1, 'direkte demokratie': 1.5},
        'French': {'élite': 1.5, 'médias traditionnels': 2, 'peuple': 1, 'corrompu': 1.5, 'discours lalbisgüetli': 1, 'mesdames messieurs': 1, 'présidente confédération': 1, 'démocratie directe': 1.5, 'madame présidente': 1, 'conseil fédéral': 2, 'droits lhomme': 1, 'citoyennes citoyens': 1, 'libre circulation': 1, 'manière dutiliser': 1},
    }

    def is_populist(text, language):
        score = 0
        for term, weight in populist_terms[language].items():
            if term in text:
                score += weight
        if score >= 5:  # Setze diesen Schwellenwert auf den gewünschten Wert
            return 1
        return 0

    df['Populism'] = df.apply(lambda row: is_populist(row['Text'], row['Language']), axis=1)

    # Speichere den bereinigten Datensatz
    df.to_csv('data/processed/processed.csv', index=False)

    return df

# Verwendung
load_and_preprocess_data('data/raw/old-newspaper.tsv')