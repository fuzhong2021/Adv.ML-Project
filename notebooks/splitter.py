import pandas as pd
from sklearn.model_selection import train_test_split

# Lies die tsv-Datei
df = pd.read_csv('/Users/furkansaygin/Documents/Studium/ML Projekt 2/data/raw/old-newspaper.tsv', sep='\t')

# Teile die Daten in 75% und 25% Teile
train, test = train_test_split(df, test_size=0.25, random_state=42)

# Speichern Sie die resultierenden DataFrames in csv-Dateien
train.to_csv('/Users/furkansaygin/Documents/Studium/ML Projekt 2/data/processed/train.csv', index=False)
test.to_csv('/Users/furkansaygin/Documents/Studium/ML Projekt 2/data/processed/test.csv', index=False)
