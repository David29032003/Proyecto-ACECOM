from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import joblib  # Para cargar el modelo guardado
import numpy as np
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Cargar los datos
ruta_datasets = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
dataset1 = pd.read_csv(os.path.join(ruta_datasets, 'dataset1.csv'))

# Crear un conjunto para géneros únicos
all_genres = set()
for genre_list in dataset1["Genre"]:
    cleaned_genres = genre_list.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
    for genre in cleaned_genres.split(","):
        all_genres.add(genre.strip().lower())

# Obtengo la ruta de la carpeta donde está ubicada la API
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo preentrenado desde el archivo .pkl
model = joblib.load(os.path.join(ruta_actual, 'sentence_transformer_model.pkl'))

# Función para extraer géneros mencionados
def extraer_generos(texto):
    texto = texto.lower()
    return [genero for genero in all_genres if genero.lower() in texto]

# Función para extraer directores y actores mencionados
def extraer_personas_mencionadas(texto):
    personas_mencionadas = set()
    for persona in ['Director', 'Star1', 'Star2', 'Star3', 'Star4']:
        for _, row in dataset1.iterrows():
            if row[persona] and row[persona].lower() in texto.lower():
                personas_mencionadas.add(row[persona])
    return list(personas_mencionadas)

# Función para buscar películas
def buscar_peliculas_con_embedding(texto):
    embeddings_texto = model.encode([texto])
    embeddings_peliculas = model.encode(dataset1['Overview'].fillna("").tolist())
    personas_mencionadas = extraer_personas_mencionadas(texto)
    similitudes = cosine_similarity(embeddings_texto, embeddings_peliculas)
    generos_texto = extraer_generos(texto)
    boost_factor = 1.5

    for idx, row in dataset1.iterrows():
        if any(persona in personas_mencionadas for persona in [row['Director']]):
            similitudes[0][idx] *= boost_factor
        movie_genres = [genre.strip().lower() for genre in row['Genre'].split(',')]
        if any(genre in generos_texto for genre in movie_genres):
            similitudes[0][idx] *= boost_factor

    indices_similares = similitudes[0].argsort()[-5:][::-1]
    peliculas_recomendadas = dataset1.iloc[indices_similares]
    return peliculas_recomendadas['Series_Title'].tolist()  # Solo devolver los títulos

# Ruta principal para la búsqueda usando el método GET
@app.route('/buscar', methods=['GET'])
def buscar():
    texto = request.args.get('texto', '')
    if not texto:
        return jsonify({"error": "Se requiere un texto para realizar la búsqueda"}), 400

    peliculas = buscar_peliculas_con_embedding(texto)

    return jsonify({"titulos": peliculas})

# Punto de entrada de la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
