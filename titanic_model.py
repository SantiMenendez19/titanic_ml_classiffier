# Practica de prediccion con arboles de clasificacion utilizando dataset del Titanic
# Fuente: https://www.kaggle.com/c/titanic

# Modulos
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pickle

import pydotplus

# Funciones genericas

def load_datasets(path_train, path_test):
    dataset_train = pd.read_csv(path_train)
    dataset_test = pd.read_csv(path_test)
    return dataset_train, dataset_test

def main():

    file_model = "model_titanic_tree.pkl"

    ### Carga del dataset de entrenamiento y test
    dataset_titanic, dataset_titanic_test = load_datasets(os.path.join(os.path.dirname(sys.argv[0]), "input", "train.csv"), 
        os.path.join(os.path.dirname(sys.argv[0]), "input", "test.csv"))

    ### Armado de los atributos

    dataset_titanic = dataset_titanic.dropna(axis=1)
    dataset_titanic_test = dataset_titanic_test.dropna(axis=1)

    X = pd.get_dummies(dataset_titanic[['Pclass', 'Sex', 'SibSp', 'Parch']])
    y = dataset_titanic['Survived']

    ### Creacion del set de entrenamiento y testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    ### Entrenamiento del modeleo

    tree_titanic = DecisionTreeClassifier()
    tree_titanic.fit(X_train, y_train)
    tree_titanic.fit(X, y)
    print("El modelo fue entrenado usando algoritmo de Clasificacion por Arboles de decision")

    # Abro un modelo ya guardado

    #with open(file_model, "rb") as file:
    #    tree_titanic = pickle.load(file)

    ### Testeo del modelo

    score = accuracy_score(y_test, tree_titanic.predict(X_test))
    print(f"Puntaje de precision: {score} ")

    ### Guardado del modelo actual

    with open(file_model, "wb") as file:
        pickle.dump(tree_titanic, file)
    print(f"El modelo fue guardado como {file_model}")

    ### Prediccion con el modelo creado

    X_test = pd.get_dummies(dataset_titanic_test[['Pclass', 'Sex', 'SibSp', 'Parch']])
    predictions = tree_titanic.predict(X_test)
    #print(predictions)

    ### Guardado de la prediccion

    output = pd.DataFrame({'PassengerId': dataset_titanic_test["PassengerId"], 'Survived': predictions})
    output.to_csv(os.path.join(os.path.dirname(sys.argv[0]), "output", "my_submission.csv"), index=False)

    ### Imprimir Arbol de Decision

    dot_data = export_graphviz(tree_titanic, out_file=None,  
                    feature_names=X_test.columns,  
                    filled=True, rounded=True,  
                    special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(os.path.join(os.path.dirname(sys.argv[0]), "tree", "decision_tree.png"))

if __name__ == "__main__":
    main()