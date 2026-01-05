from src.utils.common_funtions import read_yaml
from src.config.paths_config import *
from src.data_preprocessing.preprocessing import DataProcessor
from src.data_trainer.model_training import ModelTraining

if __name__=="__main__":
    data_processor = DataProcessor(ANIMELIST_CSV,PROCESSED_DIR)
    data_processor.run()

    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()

