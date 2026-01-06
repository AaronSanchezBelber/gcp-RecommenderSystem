# =========================
# IMPORTS
# =========================

import pandas as pd              # Para manejo de dataframes
import numpy as np               # Para operaciones matemáticas y vectores
import joblib                    # Para cargar modelos y pesos entrenados
from src.config.paths_config import *  # Importa rutas de archivos (buena práctica MLOps)


# =========================
# 1. GET_ANIME_FRAME
# =========================
# Devuelve el registro del anime desde el dataframe
# Puede buscar por ID (int) o por nombre (str)

def getAnimeFrame(anime, path_df):
    df = pd.read_csv(path_df)    # Carga el dataframe de animes

    # Si el anime se pasa como ID
    if isinstance(anime, int):
        return df[df.anime_id == anime]

    # Si el anime se pasa como nombre
    if isinstance(anime, str):
        return df[df.eng_version == anime]


# =========================
# 2. GET_SYNOPSIS
# =========================
# Devuelve la sinopsis del anime
# Puede buscar por ID o por nombre

def getSynopsis(anime, path_synopsis_df):
    synopsis_df = pd.read_csv(path_synopsis_df)  # Carga el dataframe de sinopsis

    # Búsqueda por ID
    if isinstance(anime, int):
        return synopsis_df[synopsis_df.MAL_ID == anime].sypnopsis.values[0]

    # Búsqueda por nombre
    if isinstance(anime, str):
        return synopsis_df[synopsis_df.Name == anime].sypnopsis.values[0]


# =========================
# 3. CONTENT-BASED RECOMMENDATION
# =========================
# Recomienda animes similares usando embeddings (anime-anime)

def find_similar_animes(
    name,
    path_anime_weights,
    path_anime2anime_encoded,
    path_anime2anime_decoded,
    path_anime_df,
    n=10,
    return_dist=False,
    neg=False
):
    # Carga los pesos del modelo (embeddings)
    anime_weights = joblib.load(path_anime_weights)

    # Diccionarios de codificación / decodificación
    anime2anime_encoded = joblib.load(path_anime2anime_encoded)
    anime2anime_decoded = joblib.load(path_anime2anime_decoded)

    # Obtiene el anime_id a partir del nombre
    index = getAnimeFrame(name, path_anime_df).anime_id.values[0]

    # Convierte el anime_id a índice interno del embedding
    encoded_index = anime2anime_encoded.get(index)

    # Validación por si no existe
    if encoded_index is None:
        raise ValueError(f"Encoded index not found for anime ID: {index}")

    # Calcula similitud usando producto punto
    weights = anime_weights
    dists = np.dot(weights, weights[encoded_index])

    # Ordena los índices por similitud
    sorted_dists = np.argsort(dists)

    # Se suma 1 para excluir el propio anime
    n = n + 1

    # Si neg=True devuelve los menos similares
    if neg:
        closest = sorted_dists[:n]
    else:
        closest = sorted_dists[-n:]

    # Si solo se quieren distancias
    if return_dist:
        return dists, closest

    # Construye la lista de resultados
    SimilarityArr = []

    for close in closest:
        decoded_id = anime2anime_decoded.get(close)

        anime_frame = getAnimeFrame(decoded_id, path_anime_df)

        anime_name = anime_frame.eng_version.values[0]
        genre = anime_frame.Genres.values[0]
        similarity = dists[close]

        SimilarityArr.append({
            "anime_id": decoded_id,
            "name": anime_name,
            "similarity": similarity,
            "genre": genre,
        })

    # Devuelve dataframe ordenado por similitud
    Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)

    # Elimina el anime original
    return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)


# =========================
# 4. FIND SIMILAR USERS
# =========================
# Encuentra usuarios similares usando embeddings usuario-usuario

def find_similar_users(
    item_input,
    path_user_weights,
    path_user2user_encoded,
    path_user2user_decoded,
    n=10,
    return_dist=False,
    neg=False
):
    try:
        # Carga embeddings y diccionarios
        user_weights = joblib.load(path_user_weights)
        user2user_encoded = joblib.load(path_user2user_encoded)
        user2user_decoded = joblib.load(path_user2user_decoded)

        index = item_input
        encoded_index = user2user_encoded.get(index)

        weights = user_weights

        # Similaridad por producto punto
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)

        n = n + 1

        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        if return_dist:
            return dists, closest

        SimilarityArr = []

        for close in closest:
            similarity = dists[close]

            if isinstance(item_input, int):
                decoded_id = user2user_decoded.get(close)
                SimilarityArr.append({
                    "similar_users": decoded_id,
                    "similarity": similarity
                })

        similar_users = pd.DataFrame(SimilarityArr).sort_values(
            by="similarity", ascending=False
        )

        # Elimina el propio usuario
        similar_users = similar_users[similar_users.similar_users != item_input]

        return similar_users

    except Exception as e:
        print("Error Occured", e)


# =========================
# 5. GET USER PREFERENCES
# =========================
# Obtiene los animes favoritos de un usuario
# Usa el percentil 75 de rating

def get_user_preferences(user_id, path_rating_df, path_anime_df):

    rating_df = pd.read_csv(path_rating_df)
    df = pd.read_csv(path_anime_df)

    # Filtra ratings del usuario
    animes_watched_by_user = rating_df[rating_df.user_id == user_id]

    # Calcula el percentil 75
    user_rating_percentile = np.percentile(
        animes_watched_by_user.rating, 75
    )

    # Filtra animes mejor valorados
    animes_watched_by_user = animes_watched_by_user[
        animes_watched_by_user.rating >= user_rating_percentile
    ]

    # Obtiene IDs ordenados por rating
    top_animes_user = (
        animes_watched_by_user
        .sort_values(by="rating", ascending=False)
        .anime_id.values
    )

    # Obtiene nombre y género
    anime_df_rows = df[df["anime_id"].isin(top_animes_user)]
    anime_df_rows = anime_df_rows[["eng_version", "Genres"]]

    return anime_df_rows


# =========================
# 6. USER-BASED RECOMMENDATION
# =========================
# Recomienda animes basándose en usuarios similares

def get_user_recommendations(
    similar_users,
    user_pref,
    path_anime_df,
    path_synopsis_df,
    path_rating_df,
    n=10
):

    recommended_animes = []
    anime_list = []

    # Recorre usuarios similares
    for user_id in similar_users.similar_users.values:
        pref_list = get_user_preferences(
            int(user_id), path_rating_df, path_anime_df
        )

        # Elimina animes ya vistos por el usuario original
        pref_list = pref_list[
            ~pref_list.eng_version.isin(user_pref.eng_version.values)
        ]

        if not pref_list.empty:
            anime_list.append(pref_list.eng_version.values)

    if anime_list:
        anime_list = pd.DataFrame(anime_list)

        # Cuenta frecuencia de recomendaciones
        sorted_list = (
            pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts())
            .head(n)
        )

        for anime_name in sorted_list.index:
            n_user_pref = sorted_list.loc[anime_name].values[0]

            if isinstance(anime_name, str):
                frame = getAnimeFrame(anime_name, path_anime_df)
                anime_id = frame.anime_id.values[0]
                genre = frame.Genres.values[0]
                synopsis = getSynopsis(
                    int(anime_id), path_synopsis_df
                )

                recommended_animes.append({
                    "n": n_user_pref,
                    "anime_name": anime_name,
                    "Genres": genre,
                    "Synopsis": synopsis
                })

    return pd.DataFrame(recommended_animes).head(n)
