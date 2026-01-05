import joblib
import numpy as np
import os
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping
)

from src.logger import get_logger
from src.exception.exception import CustomException
from src.base_model.base_model import BaseModel
from src.config.paths_config import *

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, data_path):
        self.data_path = data_path
        logger.info("ModelTraining initialized (NO comet_ml)")

    def load_data(self):
        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            logger.info("Data loaded successfully for training")
            return X_train_array, X_test_array, y_train, y_test
        except Exception as e:
            raise CustomException("Failed to load training data", e)

    def train_model(self):
        try:
            X_train_array, X_test_array, y_train, y_test = self.load_data()

            n_users = len(joblib.load(USER2USER_ENCODED))
            n_anime = len(joblib.load(ANIME2ANIME_ENCODED))

            base_model = BaseModel(config_path=CONFIG_PATH)
            model = base_model.RecommenderNet(
                n_users=n_users,
                n_anime=n_anime
            )

            # Learning rate schedule (FIXED)
            start_lr = 1e-5
            max_lr = 5e-5
            min_lr = 1e-6

            rampup_epochs = 5
            sustain_epochs = 0
            exp_decay = 0.8

            def lrfn(epoch):
                if epoch < rampup_epochs:
                    return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
                elif epoch < rampup_epochs + sustain_epochs:
                    return max_lr
                else:
                    return (max_lr - min_lr) * exp_decay ** (
                        epoch - rampup_epochs - sustain_epochs
                    ) + min_lr

            lr_callback = LearningRateScheduler(lrfn, verbose=0)

            checkpoint_cb = ModelCheckpoint(
                filepath=CHECKPOINT_FILE_PATH,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True
            )

            early_stopping_cb = EarlyStopping(
                patience=3,
                monitor="val_loss",
                mode="min",
                restore_best_weights=True
            )

            callbacks = [checkpoint_cb, lr_callback, early_stopping_cb]

            os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH), exist_ok=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            history = model.fit(
                x=X_train_array,
                y=y_train,
                batch_size=10000,
                epochs=20,
                validation_data=(X_test_array, y_test),
                callbacks=callbacks,
                verbose=1
            )

            model.load_weights(CHECKPOINT_FILE_PATH)
            logger.info("Model training completed successfully")

            self.save_model_weights(model)

        except Exception as e:
            logger.error(str(e))
            raise CustomException("Error during model training process", e)

    def extract_weights(self, layer_name, model):
        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
            logger.info(f"Weights extracted for layer: {layer_name}")
            return weights
        except Exception as e:
            raise CustomException("Error during weight extraction", e)

    def save_model_weights(self, model):
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved at {MODEL_PATH}")

            user_weights = self.extract_weights("user_embedding", model)
            anime_weights = self.extract_weights("anime_embedding", model)

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            logger.info("User & Anime weights saved successfully")

        except Exception as e:
            raise CustomException("Error while saving model and weights", e)


if __name__ == "__main__":
    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()
