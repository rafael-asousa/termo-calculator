import pandas as pd

df1 = pd.read_csv('ranking_1602_useful.csv')

print(df1.sort_values('entropy'))