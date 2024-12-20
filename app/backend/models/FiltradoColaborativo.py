# Importo las librerías
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from ImportacionDataset2 import dataset2

rating_matrix = dataset2.pivot(index='id_usuario', columns='id_pelicula', values='rating')

rating_matrix = rating_matrix.fillna(0)

diccionario_pelicula = dict(zip(dataset2['id_pelicula'], dataset2['titulo_pelicula']))

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(rating_matrix)

# Defino una función para realizar recomendaciones
def recomendaciones_peliculas(id_usuario, cantidad_recomendaciones=5):

    # Obtengo la fila del usuario en la matriz
    usuario_vector = rating_matrix.loc[id_usuario].values.reshape(1, -1)

    # Encuentro los usuarios más cercanos
    distancias, indices = knn.kneighbors(usuario_vector, n_neighbors=cantidad_recomendaciones + 1)

    # Excluyo al propio usuario
    usuarios_similares = indices.flatten()[1:]
    distancias_similares = distancias.flatten()[1:]

    # Combino las recomendaciones de los usuarios similares
    peliculas_recomendadas = {}
    # Obtengo las películas vistas por el usuario
    peliculas_vistas = rating_matrix.loc[id_usuario][rating_matrix.loc[id_usuario] > 0].index

    for id_similar, distancia in zip(usuarios_similares, distancias_similares):
        id_usuario_similar = rating_matrix.index[id_similar]
        peliculas_similares = rating_matrix.loc[id_usuario_similar][rating_matrix.loc[id_usuario_similar] > 0]

        for id_pelicula, rating in peliculas_similares.items():
            # Solo recomiendo películas no vistas por el usuario
            if id_pelicula not in peliculas_vistas and id_pelicula not in peliculas_recomendadas:
                peliculas_recomendadas[id_pelicula] = rating / distancia  # Pondero por la distancia

    # Ordeno las películas recomendadas por la puntuación calculada
    peliculas_ordenadas = sorted(peliculas_recomendadas.items(), key=lambda x: x[1], reverse=True)

    # Devuelvo las N mejores películas
    mejores_peliculas = [pelicula[0] for pelicula in peliculas_ordenadas[:cantidad_recomendaciones]]

    # Devuelvo las N mejores películas con sus títulos
    mejores_peliculas = [diccionario_pelicula[pelicula[0]] for pelicula in peliculas_ordenadas[:cantidad_recomendaciones] if pelicula[0] in diccionario_pelicula]

    return mejores_peliculas

# Ejemplo de uso
usuario_ejemplo = 42
recomendaciones = recomendaciones_peliculas(usuario_ejemplo, cantidad_recomendaciones=5)
print(f"Películas recomendadas para el usuario {usuario_ejemplo}: {recomendaciones}")
