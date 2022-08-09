import pandas as pd

df = pd.read_csv('nba.csv')

print(df) 

print('----------------------------')

print(df.head())
print(df.tail())
print(df.info())