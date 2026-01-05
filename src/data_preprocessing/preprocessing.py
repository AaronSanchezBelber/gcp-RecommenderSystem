import pandas as pd
import os 
import numpy as np
import joblib
import sys

from src.logger.logger import get_logger 
from src.exception.exception import CustomException
from src.config.paths_config import *

# Se crea un logger para registrar mensajes de ejecución (info, errores, etc.)
logger = get_logger(__name__)


# Clase encargada de todo el preprocesamiento de datos
class DataProcessor:
    def __init__(self, input_file, output_dir):
        # Ruta del CSV de ratings (user_id, anime_id, rating)
        self.input_file = input_file

        # Carpeta donde se guardarán los datos procesados
        self.output_dir = output_dir

        # DataFrames principales (se inicializan vacíos)
        self.rating_df = None
        self.anime_df = None

        # Datos para entrenamiento y test
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        # Diccionarios de codificación (ID real ↔ índice)
        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}

        # Crea el directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("DataProcessing Initialized")
    
    # -------------------- CARGA DE DATOS --------------------
    def load_data(self, usecols):
        try:
            # Carga el CSV solo con las columnas necesarias
            self.rating_df = pd.read_csv(
                self.input_file,
                low_memory=True,
                usecols=usecols
            )
            logger.info("Data loaded successfully for Data Processing")
        except Exception as e:
            # Lanza excepción personalizada si falla
            raise CustomException("Failed to load data", sys)
        
    # -------------------- FILTRADO DE USUARIOS --------------------
    def filter_users(self, min_rating=400):
        try:
            # Cuenta cuántos ratings ha hecho cada usuario
            n_ratings = self.rating_df["user_id"].value_counts()

            # Se queda solo con usuarios con al menos min_rating valoraciones
            self.rating_df = self.rating_df[
                self.rating_df["user_id"].isin(
                    n_ratings[n_ratings >= min_rating].index
                )
            ].copy()

            logger.info("Filtered users successfully...")
        except Exception as e:
            raise CustomException("Failed to filter data", sys)
    
    # -------------------- ESCALADO DE RATINGS --------------------
    def scale_ratings(self):
        try:
            # Obtiene el mínimo y máximo rating
            min_rating = min(self.rating_df["rating"])
            max_rating = max(self.rating_df["rating"])

            # Normaliza los ratings a rango [0,1]
            self.rating_df["rating"] = (
                self.rating_df["rating"]
                .apply(lambda x: (x - min_rating) / (max_rating - min_rating))
                .values
                .astype(np.float64)
            )

            logger.info("Scaling done for Processing")
        except Exception as e:
            raise CustomException("Failed to scale data", sys)
    
    # -------------------- CODIFICACIÓN DE USUARIOS Y ANIMES --------------------
    def encode_data(self):
        try:
            ### -------- USERS --------
            # Obtiene IDs únicos de usuarios
            user_ids = self.rating_df["user_id"].unique().tolist()

            # Crea diccionarios de codificación
            self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            self.user2user_decoded = {i: x for i, x in enumerate(user_ids)}

            # Añade columna "user" con IDs codificados
            self.rating_df["user"] = self.rating_df["user_id"].map(
                self.user2user_encoded
            )

            ### -------- ANIME --------
            anime_ids = self.rating_df["anime_id"].unique().tolist()

            self.anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
            self.anime2anime_decoded = {i: x for i, x in enumerate(anime_ids)}

            # Añade columna "anime" con IDs codificados
            self.rating_df["anime"] = self.rating_df["anime_id"].map(
                self.anime2anime_encoded
            )

            logger.info("Encoding done for Users and Anime")
        except Exception as e:
            raise CustomException("Failed to encode data", sys)
    
    # -------------------- TRAIN / TEST SPLIT --------------------
    def split_data(self, test_size=1000, random_state=43):
        try:
            # Baraja aleatoriamente el dataset
            self.rating_df = (
                self.rating_df
                .sample(frac=1, random_state=random_state)
                .reset_index(drop=True)
            )

            # Variables de entrada (user, anime)
            X = self.rating_df[["user", "anime"]].values

            # Variable objetivo (rating normalizado)
            y = self.rating_df["rating"]

            # Índice de corte train / test
            train_indices = self.rating_df.shape[0] - test_size

            # División manual
            X_train, X_test, y_train, y_test = (
                X[:train_indices],
                X[train_indices:],
                y[:train_indices],
                y[train_indices:],
            )

            # Formato que espera Keras: [users, animes]
            self.X_train_array = [X_train[:, 0], X_train[:, 1]]
            self.X_test_array = [X_test[:, 0], X_test[:, 1]]
            self.y_train = y_train
            self.y_test = y_test

            logger.info("Data split successfully")
        except Exception as e:
            raise CustomException("Failed to split data", sys)
    
    # -------------------- GUARDADO DE ARTEFACTOS --------------------
    def save_artifacts(self):
        try:
            # Diccionarios a persistir
            artifacts = {
                "user2user_encoded": self.user2user_encoded,
                "user2user_decoded": self.user2user_decoded,
                "anim2anime_encoded": self.anime2anime_encoded,
                "anim2anime_decoded": self.anime2anime_decoded,
            }

            # Guarda cada diccionario como .pkl
            for name, data in artifacts.items():
                joblib.dump(
                    data,
                    os.path.join(self.output_dir, f"{name}.pkl")
                )
                logger.info(f"{name} saved successfully")

            # Guarda datos de entrenamiento y test
            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)

            # Guarda el dataframe final de ratings
            self.rating_df.to_csv(RATING_DF, index=False)

            logger.info("All processed data saved successfully")
        except Exception as e:
            raise CustomException("Failed to save artifacts data", sys)
        
    # -------------------- PROCESAMIENTO DE DATOS DE ANIME --------------------
    def process_anime_data(self):
        try:
            # Carga el dataset principal de anime
            df = pd.read_csv(ANIME_CSV)

            # Columnas relevantes para sinopsis
            cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
            synopsis_df = pd.read_csv(
                ANIMESYNOPSIS_CSV,
                usecols=cols
            )

            # Reemplaza "Unknown" por NaN
            df = df.replace("Unknown", np.nan)

            # Función auxiliar para obtener el nombre del anime
            def getAnimeName(anime_id):
                try:
                    name = df[df.anime_id == anime_id].eng_version.values[0]
                    if name is np.nan:
                        name = df[df.anime_id == anime_id].Name.values[0]
                except:
                    print("Error")
                return name
                
            # Unifica nombres de columnas
            df["anime_id"] = df["MAL_ID"]
            df["eng_version"] = df["English name"]

            # Obtiene el nombre correcto del anime
            df["eng_version"] = df.anime_id.apply(
                lambda x: getAnimeName(x)
            )

            # Ordena por score de MyAnimeList
            df.sort_values(
                by=["Score"],
                inplace=True,
                ascending=False,
                kind="quicksort",
                na_position="last"
            )

            # Selecciona columnas finales
            df = df[
                ["anime_id", "eng_version", "Score",
                 "Genres", "Episodes", "Type",
                 "Premiered", "Members"]
            ]

            # Guarda los datasets procesados
            df.to_csv(DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)

            logger.info("DF and SYNOPSIS_DF saved successfully")

        except Exception as e:
            raise CustomException("Failed to process anime data", sys)
    
    # -------------------- PIPELINE COMPLETO --------------------
    def run(self):
        try:
            self.load_data(usecols=["user_id", "anime_id", "rating"])
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()

            logger.info("Data Processing Pipeline ran successfully")
        except CustomException as e:
            logger.error(str(e))


# -------------------- EJECUCIÓN DEL SCRIPT --------------------
if __name__ == "__main__":
    # Se instancia el procesador con rutas de entrada y salida
    data_processor = DataProcessor(
        ANIMELIST_CSV,
        PROCESSED_DIR
    )

    # Se ejecuta todo el pipeline
    data_processor.run()
