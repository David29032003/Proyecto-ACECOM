import os
import pandas as pd

ruta = os.path.expanduser("~\\Downloads\\Proyecto-ACECOM\\app\\backend\\datasets\\dataset2.csv")

# Cargar el dataset
dataset2 = pd.read_csv(ruta)

print(dataset2.head())

print(dataset2.shape)