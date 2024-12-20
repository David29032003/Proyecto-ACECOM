from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from flask_cors import CORS

# --- Carga ---

# Definición de rutas de datasets y modelos
ruta_carpeta = os.path.dirname(os.path.abspath(__file__))
ruta_datasets = os.path.join(ruta_carpeta, '..', 'datasets')

# Cargo los datasets
dataset1 = pd.read_csv(os.path.join(ruta_datasets, 'dataset1.csv'))
dataset2 = pd.read_csv(os.path.join(ruta_datasets, 'dataset2.csv'))

# Inicio una instancia de Flask
app = Flask(__name__)
CORS(app)

# Cargo los modelos preentrenados
sentence_transformer = joblib.load(os.path.join(ruta_carpeta, 'sentence_transformer_model.pkl'))
similitud_coseno = joblib.load(os.path.join(ruta_carpeta, 'similitud_coseno.pkl'))
knn = joblib.load(os.path.join(ruta_carpeta, 'knn_model.pkl'))

# Creo un conjunto de géneros únicos
generos = set()
for lista_genero in dataset1["Genre"]:
    generos_limpios = lista_genero.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
    for genre in generos_limpios.split(","):
        generos.add(genre.strip().lower())

# Diccionario para títulos de películas en dataset2
diccionario_pelicula = dict(zip(dataset2['id_pelicula'], dataset2['titulo_pelicula']))

# --- Funciones ---

def obtener_recomendaciones_contenido(titulo):
    indice = dataset1[dataset1['Series_Title'] == titulo].index[0]
    puntajes_similitud = list(enumerate(similitud_coseno[indice]))
    puntajes_similitud = sorted(puntajes_similitud, key=lambda x: x[1], reverse=True)
    indices_similares = [i[0] for i in puntajes_similitud[1:6]]
    return dataset1['Series_Title'].iloc[indices_similares].tolist()

def obtener_recomendaciones_colaborativo(id_usuario, cantidad_recomendaciones=5):
    rating_matrix = dataset2.pivot(index='id_usuario', columns='id_pelicula', values='rating').fillna(0)
    usuario_vector = rating_matrix.loc[id_usuario].values.reshape(1, -1)
    distancias, indices = knn.kneighbors(usuario_vector, n_neighbors=cantidad_recomendaciones + 1)
    usuarios_similares = indices.flatten()[1:]
    distancias_similares = distancias.flatten()[1:]

    peliculas_recomendadas = {}
    peliculas_vistas = rating_matrix.loc[id_usuario][rating_matrix.loc[id_usuario] > 0].index

    for id_similar, distancia in zip(usuarios_similares, distancias_similares):
        id_usuario_similar = rating_matrix.index[id_similar]
        peliculas_similares = rating_matrix.loc[id_usuario_similar][rating_matrix.loc[id_usuario_similar] > 0]
        for id_pelicula, rating in peliculas_similares.items():
            if id_pelicula not in peliculas_vistas and id_pelicula not in peliculas_recomendadas:
                peliculas_recomendadas[id_pelicula] = rating / distancia

    peliculas_ordenadas = sorted(peliculas_recomendadas.items(), key=lambda x: x[1], reverse=True)
    mejores_peliculas = [diccionario_pelicula[pelicula[0]] for pelicula in peliculas_ordenadas[:cantidad_recomendaciones] if pelicula[0] in diccionario_pelicula]

    return mejores_peliculas

def obtener_generos(texto):
    texto = texto.lower()
    return [genero for genero in generos if genero in texto]

def obtener_personas_mencionadas(texto):
    personas_mencionadas = set()
    for persona in ['Director', 'Star1', 'Star2', 'Star3', 'Star4']:
        for _, fila in dataset1.iterrows():
            if fila[persona] and fila[persona].lower() in texto.lower():
                personas_mencionadas.add(fila[persona])
    return list(personas_mencionadas)

def buscar_peliculas_con_embedding(texto):
    embeddings_texto = sentence_transformer.encode([texto])
    embeddings_peliculas = sentence_transformer.encode(dataset1['Overview'].fillna("").tolist())
    personas_mencionadas = obtener_personas_mencionadas(texto)
    similitudes = cosine_similarity(embeddings_texto, embeddings_peliculas)
    generos_texto = obtener_generos(texto)
    factor = 1.5

    for id, fila in dataset1.iterrows():

        if fila['Director'] in personas_mencionadas:
            similitudes[0][id] *= factor
        movie_genres = [genre.strip().lower() for genre in fila['Genre'].split(',')]
        
        if any(genre in generos_texto for genre in movie_genres):
            similitudes[0][id] *= factor

    indices_similares = similitudes[0].argsort()[-5:][::-1]
    peliculas_recomendadas = dataset1.iloc[indices_similares]
    return peliculas_recomendadas['Series_Title'].tolist()

# --- Endpoints ---

@app.route('/contenido', methods=['GET'])
def contenido():

    titulo = request.args.get('titulo')

    if not titulo:
        return jsonify({'error': 'Proporcione un titulo de pelicula'}), 400
    recomendaciones = obtener_recomendaciones_contenido(titulo)

    if not recomendaciones:
        return jsonify({'error': 'No se encontraron recomendaciones para este titulo'}), 404
    return jsonify({'recomendaciones': recomendaciones})

@app.route('/colaborativo', methods=['GET'])
def colaborativo():

    id_usuario = request.args.get('id_usuario', type=int)

    if not id_usuario:
        return jsonify({"error": "Proporcione un id de usuario"}), 400
    recomendaciones = obtener_recomendaciones_colaborativo(id_usuario)

    if not recomendaciones:
        return jsonify({'error': 'No se encontraron recomendaciones para este usuario'}), 404
    return jsonify({"usuario": id_usuario, "recomendaciones": recomendaciones})

@app.route('/buscar', methods=['GET'])
def buscar():

    texto = request.args.get('texto', '')

    if not texto:
        return jsonify({"error": "Se requiere un texto para realizar la búsqueda"}), 400
    peliculas = buscar_peliculas_con_embedding(texto)
    return jsonify({"titulos": peliculas})

# --- Ejecución de la aplicación ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
