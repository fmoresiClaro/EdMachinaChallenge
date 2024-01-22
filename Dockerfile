# Usar una imagen base de Python
FROM python:3.11-slim-buster

# Establecer un directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar la aplicación
CMD ["python", "EdMachina1.py"]

