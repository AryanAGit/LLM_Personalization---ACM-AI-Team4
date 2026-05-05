import pandas as pd
import json

df = pd.read_csv(
    'trump_data/realDonaldTrump_in_office.csv',
    on_bad_lines='skip',
    skipinitialspace=True,
)


df = df[~df['Tweet Text'].str.contains(r'https?://', na=False, regex=True)][['Tweet Text']]
df = df.head(1500)

data = [{"prompt": "", "output": text} for text in df['Tweet Text'].dropna()]

with open('trump_data/trump_tweets_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)