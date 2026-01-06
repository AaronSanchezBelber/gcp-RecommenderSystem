### Accdeso a GCP

(venv) C:\Users\Lenovo\Desktop\krish-udemy\mlops-recomenderSystem-2026>set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Lenovo\Downloads\recommerder-153c2ed22467.json

### Workflow

0) Utils
1) config config.yaml 
    Data Ingestion para bucket
2) config paths_config.py 

python -m src.data_ingestion.ingestion (Ejecutas un módulo dentro de un paquete)

ingestion o preproceso. hacer la clase en el .py de src. luego los paths en config

python -m src.data_preprocessing.preprocessing

python -m src.data_trainer.model_training

git ls-files -z | xargs -0 du -h | sort -h
(por si tienes problemas con el git por mter datos)

GCP
habilitar 
google cloud registry
google cloud artifacts
kubernetes engine api

una vez la ci-cd pipeline esta hecha vas a GcP y revisas:
- Container registry -ml project (la imagen esta bien)
- Kuberbets Engine - workflow- ml app- puedes mirar los puertos que estan corriendo - ENDPOINT