import pandas as pd

df = pd.read_csv('pokemon_data.csv')

#print(df[['Name', 'HP', 'Attack']])
#print(df.iloc[3][2])

#for index, row in df.iterrows():
#    print(index, row[['Name', 'HP']])

#print(df.loc[df['Type 1'] == "Grass"])
print(df.describe())