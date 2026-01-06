# =========================
# IMPORTS
# =========================

from src.config.paths_config import *     # Rutas a datasets, modelos y pesos
from src.utils.helpers import *           # Funciones auxiliares (user/content-based)


# =========================
# HYBRID RECOMMENDATION
# =========================
# Combina recomendaciones basadas en usuarios + contenido
# user_weight y content_weight controlan la importancia de cada enfoque

def hybrid_recommendation(user_id, user_weight=0.5, content_weight=0.5):

    # =========================
    # 1. USER-BASED RECOMMENDATION
    # =========================

    # Encuentra usuarios similares al usuario objetivo
    similar_users = find_similar_users(
        user_id,
        USER_WEIGHTS_PATH,
        USER2USER_ENCODED,
        USER2USER_DECODED
    )

    # Obtiene las preferencias (animes mejor valorados) del usuario
    user_pref = get_user_preferences(
        user_id,
        RATING_DF,
        DF
    )

    # Obtiene recomendaciones basadas en usuarios similares
    user_recommended_animes = get_user_recommendations(
        similar_users,
        user_pref,
        DF,
        SYNOPSIS_DF,
        RATING_DF
    )

    # Convierte el dataframe de recomendaciones en una lista de nombres
    user_recommended_anime_list = (
        user_recommended_animes["anime_name"].tolist()
    )


    # =========================
    # 2. CONTENT-BASED RECOMMENDATION
    # =========================

    # Lista para almacenar recomendaciones basadas en contenido
    content_recommended_animes = []

    # Para cada anime recomendado por usuarios similares
    for anime in user_recommended_anime_list:

        # Busca animes similares usando embeddings de contenido
        similar_animes = find_similar_animes(
            anime,
            ANIME_WEIGHTS_PATH,
            ANIME2ANIME_ENCODED,
            ANIME2ANIME_DECODED,
            DF
        )

        # Si se encuentran animes similares, se agregan a la lista
        if similar_animes is not None and not similar_animes.empty:
            content_recommended_animes.extend(
                similar_animes["name"].tolist()
            )
        else:
            # Mensaje informativo si no hay similitudes
            print(f"No similar anime found {anime}")


    # =========================
    # 3. COMBINACIÓN DE SCORES
    # =========================

    # Diccionario para acumular puntuaciones finales
    combined_scores = {}

    # Suma el peso de user-based recommendation
    for anime in user_recommended_anime_list:
        combined_scores[anime] = (
            combined_scores.get(anime, 0) + user_weight
        )

    # Suma el peso de content-based recommendation
    for anime in content_recommended_animes:
        combined_scores[anime] = (
            combined_scores.get(anime, 0) + content_weight
        )

    # Ordena los animes por score combinado (descendente)
    sorted_animes = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Devuelve el top 10 de animes recomendados
    return [anime for anime, score in sorted_animes[:10]]
