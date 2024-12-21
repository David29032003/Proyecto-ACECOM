import joblib
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets')))

# Guardo la ruta absoluta en una variable
ruta_datasets = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

# Cargar los datasets directamente desde los archivos CSV
dataset1 = pd.read_csv(os.path.join(ruta_datasets, 'dataset1.csv'))
dataset2 = pd.read_csv(os.path.join(ruta_datasets, 'dataset2.csv'))

# Creo la nueva columna 'contenido' para cocatenar la información relevante de las películas
# en un texto
dataset1['contenido'] = (
    dataset1['Genre'].apply(lambda x: ' '.join(x)) + ' ' +
    dataset1['Overview'] + ' ' +
    dataset1['Star1'] + ' ' +
    dataset1['Star2'] + ' ' +
    dataset1['Star3'] + ' ' +
    dataset1['Star4']
)

# Convierto la columna 'contenido' a tipo string
dataset1['contenido'] = dataset1['contenido'].astype(str)

# Cargo el modelo de embeddings preentrenado
modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
embedding = modelo_embedding.encode(dataset1['contenido'].tolist(), show_progress_bar=True)

# Obtengo la ruta de la carpeta actual
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Guardo el modelo preentrenado en un archivo .pkl
joblib.dump(embedding, os.path.join(ruta_actual, 'embedding.pkl'))

# Calculo y guardo la matriz de similitudes coseno
matriz_similitud = cosine_similarity(embedding, embedding)
joblib.dump(matriz_similitud, os.path.join(ruta_actual, 'matriz_similitud.pkl'))

# Preparo y entreno el modelo KNN
matriz_rating = dataset2.pivot(index='id_usuario', columns='id_pelicula', values='rating').fillna(0)
knn = NearestNeighbors(metric='euclidean', algorithm='ball_tree')
knn.fit(matriz_rating)

# Guardo modelo KNN entrenado
joblib.dump(knn, os.path.join(ruta_actual, 'modelo_knn.pkl'))

# Cargo el modelo embedding preentrenado para el buscador
modelo_buscador = SentenceTransformer('all-MiniLM-L6-v2')

# Guardo el modelo preentrenado en un archivo .pkl
joblib.dump(modelo_buscador, os.path.join(ruta_actual, 'modelo_buscador.pkl'))