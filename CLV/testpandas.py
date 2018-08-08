import pandas as pd

d = {'Hanoi': 29, 'Hung Yen': 89, 'Hai Duong': 34, 'Hai Phong': 15,

     'TPHCM': 52, 'Bac Ninh': 99, 'Bac Giang': 98, 'Da Nang': 43, 'Nam Dinh':18, 'Thai Binh': 17}

cities = pd.Series(d)
#print(cities)
#print(cities.head())
#print(cities.index)
#print(cities.describe())
#print(cities.sum())
#print(cities.iloc[5])
print(cities[cities > 50])