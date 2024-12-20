import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from ImportacionDataset3 import dataset3
import ast

# Creo la columna contenido
dataset3['contenido'] = (
    dataset3['Genre'].apply(lambda x: ' '.join(x)) + ' ' +
    dataset3['Overview'] + ' ' +
    dataset3['Star1'] + ' ' +
    dataset3['Star2'] + ' ' +
    dataset3['Star3'] + ' ' +
    dataset3['Star4']
)

# Convierto la columna 'contenido' a tipo string, reemplazando NaN por una cadena vacía
dataset3['contenido'] = dataset3['contenido'].fillna('').astype(str)

# Cargo el modelo de embeddings
modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
embedding = modelo_embedding.encode(dataset3['contenido'].tolist(), show_progress_bar=True)
similitud_coseno = cosine_similarity(embedding, embedding)

def obtener_recomendaciones3(titulo, similitud=similitud_coseno):
    indice = dataset3[dataset3['Series_Title'] == titulo].index[0]
    puntajes_similitud = list(enumerate(similitud[indice]))
    puntajes_similitud = sorted(puntajes_similitud, key=lambda x: x[1], reverse=True)
    indices_similares = [i[0] for i in puntajes_similitud[1:11]]
    return dataset3['Series_Title'].iloc[indices_similares]

titulo_pelicula = "Drishyam"
recomendaciones = obtener_recomendaciones3(titulo_pelicula)

print(f"Películas recomendadas para '{titulo_pelicula}':")
print(recomendaciones)



