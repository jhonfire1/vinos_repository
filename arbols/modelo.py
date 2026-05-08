import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt


wine=load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Predicciones:", accuracy_score(y_test, y_pred))  
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("")
plt.figure(figsize=(12, 8))
tree.plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=wine.target_names, 
    filled=True
    )
plt.show()
