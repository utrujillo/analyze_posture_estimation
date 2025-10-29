# Configuraciones necesarias para la instalacion de MediaPipev0.10

### 🚀 Instalacion de MediaPipev0.10 utilizando Conda

> La instalacion parte de que ya se tiene instalado Anaconda en el computador, puede descargarse del siguiente sitio web <a href="https://www.anaconda.com/download/success">Conda</a>

**1. Creacion de entorno virutal**
```
conda create -p ./mpipe_env python=3.12.0 -y
conda activate ./mpipe_env
```

**2. Instalar a partir del archivo requirements.txt**
```
pip install -r requirements.txt
```

> Es importante instalar pandas e ipynerkel al final ya que la compilacion de las librerias de mmcv-lite requeiren la version de numpy 1.2.4, pero al instalr pandas, esta version de numpy se actualiza a una mas reciente ej. 2.2.6

**3. Reinstalar numpy a la version que ocupa mmpose**
```
pip uninstall numpy -y
pip install numpy==1.26.4
```

> Al finalizar todas las instalaciones, la version de numpy que debe quedar instalada es la 1.2.4
Verificar la version de numpy instalada `pip list | grep numpy`

