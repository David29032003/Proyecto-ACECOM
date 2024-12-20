import os
import pandas as pd

ruta = os.path.expanduser("~\\Downloads\\Proyecto-ACECOM\\app\\backend\\datasets\\dataset1.csv")

# Cargar el dataset
dataset1 = pd.read_csv(ruta)


print(dataset1.head())

print(dataset1.shape)