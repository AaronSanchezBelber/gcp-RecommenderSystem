# Usamos una imagen base ligera de Python 3.8
FROM python:3.8-slim

# Variables de entorno:
# - PYTHONDONTWRITEBYTECODE evita que Python genere archivos .pyc
# - PYTHONUNBUFFERED hace que los logs se muestren en tiempo real (útil en Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalación de dependencias del sistema necesarias para librerías
# como TensorFlow, NumPy, SciPy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \        
    libatlas-base-dev \      
    libhdf5-dev \            
    libprotobuf-dev \        
    protobuf-compiler \    
    python3-dev \           
    && apt-get clean \       
    && rm -rf /var/lib/apt/lists/*  

# Definimos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos todo el código del proyecto al contenedor
COPY . .

# Instalamos las dependencias del proyecto
# -e . indica instalación editable (útil en desarrollo)
# --no-cache-dir reduce el tamaño de la imagen
RUN pip install --no-cache-dir -e .

# Ejecutamos el pipeline de entrenamiento del modelo
# Esto entrena el modelo durante el build de la imagen
RUN python pipeline/training_pipeline.py

# Exponemos el puerto donde correrá Flask
EXPOSE 5000

# Comando por defecto para ejecutar la aplicación Flask
CMD ["python", "app.py"]
