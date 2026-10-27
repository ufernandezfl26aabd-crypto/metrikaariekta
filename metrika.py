import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("etxeak.csv")
X = df[["prezioa", "azalera"]]
y = df["logelak"]

modelo = KNeighborsRegressor(n_neighbors=3)
modelo.fit(X, y)

print(r2_score(y, modelo.predict(X)))
