import joblib
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import sys

# Ajusta sys.path para incluir la carpeta 'datasets' que está fuera de 'ApiRest'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets')))

# Definir la ruta de los datasets
ruta_datasets = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

# Cargar los datasets directamente desde los archivos CSV
dataset1 = pd.read_csv(os.path.join(ruta_datasets, 'dataset1.csv'))
dataset2 = pd.read_csv(os.path.join(ruta_datasets, 'dataset2.csv'))

# Preparo el dataset para embeddings
dataset1['contenido'] = (
    dataset1['Genre'].apply(lambda x: ' '.join(x)) + ' ' +
    dataset1['Overview'] + ' ' +
    dataset1['Star1'] + ' ' +
    dataset1['Star2'] + ' ' +
    dataset1['Star3'] + ' ' +
    dataset1['Star4']
)

# Convierto la columna 'contenido' a tipo string, reemplazando NaN por una cadena vacía
dataset1['contenido'] = dataset1['contenido'].fillna('').astype(str)

# Cargo el modelo de embeddings
modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
embedding = modelo_embedding.encode(dataset1['contenido'].tolist(), show_progress_bar=True)

# Obtengo la ruta de la carpeta actual
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Guardo los embeddings y modelos en la misma carpeta que la API
joblib.dump(embedding, os.path.join(ruta_actual, 'embedding.pkl'))

# Calculo y guardo similitud coseno
similitud_coseno = cosine_similarity(embedding, embedding)
joblib.dump(similitud_coseno, os.path.join(ruta_actual, 'similitud_coseno.pkl'))

# Preparo y entreno el modelo KNN
rating_matrix = dataset2.pivot(index='id_usuario', columns='id_pelicula', values='rating').fillna(0)
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(rating_matrix)

# Guardo modelo KNN
joblib.dump(knn, os.path.join(ruta_actual, 'knn_model.pkl'))

# Cargar el modelo preentrenado
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Guardar el modelo en un archivo .pkl
joblib.dump(model, os.path.join(ruta_actual, 'sentence_transformer_model.pkl'))