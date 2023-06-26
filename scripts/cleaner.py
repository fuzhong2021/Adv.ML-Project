import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

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
    # Angenommen, 'irrelevant_feature1' und 'irrelevant_feature2' sind die Merkmale, die du entfernen möchtest.
    if 'irrelevant_feature1' in df.columns and 'irrelevant_feature2' in df.columns:
        df = df.drop(columns=['irrelevant_feature1', 'irrelevant_feature2'])

    # Speichern Sie den bereinigten Datensatz
    df.to_csv('data/processed/processed.csv', index=False)

    return df

# Verwendung
load_and_preprocess_data('data/raw/old-newspaper.tsv')
