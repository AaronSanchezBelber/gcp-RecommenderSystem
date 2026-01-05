import os
from pathlib import Path

# Nombre del proyecto (puedes cambiarlo)
PROJECT_NAME = "ml_project"

# Lista de carpetas a crear
list_of_folders = [
    f"{PROJECT_NAME}/src",
    f"{PROJECT_NAME}/src/logger",
    f"{PROJECT_NAME}/src/exception",
    f"{PROJECT_NAME}/src/data_ingestion",
    f"{PROJECT_NAME}/src/data_preprocessing",
    f"{PROJECT_NAME}/src/pipeline",
    f"{PROJECT_NAME}/src/config",
    f"{PROJECT_NAME}/src/artifacts",
    f"{PROJECT_NAME}/notebooks",
    f"{PROJECT_NAME}/app",
    f"{PROJECT_NAME}/app/templates",
    f"{PROJECT_NAME}/app/static",
    f"{PROJECT_NAME}/docker",
    f"{PROJECT_NAME}/tests",
    f"{PROJECT_NAME}/logs",
]

# Lista de archivos a crear
list_of_files = [
    f"{PROJECT_NAME}/requirements.txt",
    f"{PROJECT_NAME}/setup.py",
    f"{PROJECT_NAME}/README.md",
    f"{PROJECT_NAME}/Jenkinsfile",
    f"{PROJECT_NAME}/docker/Dockerfile",

    f"{PROJECT_NAME}/src/__init__.py",

    f"{PROJECT_NAME}/src/logger/__init__.py",
    f"{PROJECT_NAME}/src/logger/logger.py",

    f"{PROJECT_NAME}/src/exception/__init__.py",
    f"{PROJECT_NAME}/src/exception/exception.py",

    f"{PROJECT_NAME}/src/data_ingestion/__init__.py",
    f"{PROJECT_NAME}/src/data_ingestion/ingestion.py",

    f"{PROJECT_NAME}/src/data_preprocessing/__init__.py",
    f"{PROJECT_NAME}/src/data_preprocessing/preprocessing.py",

    f"{PROJECT_NAME}/src/pipeline/__init__.py",

    f"{PROJECT_NAME}/src/config/__init__.py",
    f"{PROJECT_NAME}/src/config/config.yaml",
    f"{PROJECT_NAME}/src/config/paths_config.py",
]

def create_project_structure():
    print("🚀 Creando estructura del proyecto...")

    # Crear carpetas
    for folder in list_of_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Carpeta creada: {folder_path}")

    # Crear archivos
    for file in list_of_files:
        file_path = Path(file)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            print(f"📄 Archivo creado: {file_path}")

    print("✅ Estructura del proyecto creada correctamente")

if __name__ == "__main__":
    create_project_structure()
