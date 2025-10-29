# Configuraciones necesarias para las herramientas de apoyo

### 🚀 Se crean 2 scripts
1. image_resizer_tool.py, Se encarga de cambiar el tamaño de las imagenes
2. video_resizer_tool.py, Se encarga de cambiar el tamaño de los videos

> La instalacion parte de que ya se tiene instalado Anaconda en el computador, puede descargarse del siguiente sitio web <a href="https://www.anaconda.com/download/success">Conda</a>

**1. Creacion de entorno virutal**
```
conda create -p ./utils_env python=3.12.0 -y
conda activate ./utils_env
```

**2. Instalar a partir del archivo requirements.txt**
```
pip install -r requirements.txt
```