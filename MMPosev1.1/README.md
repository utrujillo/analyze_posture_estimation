# Configuraciones necesarias para la instalacion de MMPose

### 🚀 Instalacion de MMPose utilizando Conda

> La instalacion parte de que ya se tiene instalado Anaconda en el computador, puede descargarse del siguiente sitio web <a href="https://www.anaconda.com/download/success">Conda</a>

**1. Creacion de entorno virutal**
```
conda create -p ./mmpose_env python=3.10 -y
conda activate ./mmpose_env
```

**2. Instalacion de dependencias estandar**
```
conda install pytorch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1 cpuonly -c pytorch -y
pip install opencv-python==4.11.0.86 matplotlib==3.8.4 numpy==1.24.3
```

**3. Instalacion de dependencias para el funcionamiento de MMPose**
```
pip install mmengine==0.7.4
pip install mmdet==3.1.0
pip install mmcv-lite==2.0.1
pip install mmpose==1.3.2
```

> Es importante instalar la version mmcv-lite, ya que no se va a utilizar la GPU, si se instala la version mmcv normal, se debe compilar y muchas veces da error, ya que depende de la version del SO instalado, y algunas librerias de C++, es mas problema

**4. Instalar otros paquetes necesarios para las pruebas**
```
pip install pandas openpyxl seaborn python-dotenv tqdm ipykernel
```

> Es importante instalar pandas e ipynerkel al final ya que la compilacion de las librerias de mmcv-lite requeiren la version de numpy 1.2.4, pero al instalr pandas, esta version de numpy se actualiza a una mas reciente ej. 2.2.6

**5. Reinstalar numpy a la version que ocupa mmpose**
```
pip uninstall numpy -y
pip install numpy==1.24.4
```
> Al finalizar todas las instalaciones, la version de numpy que debe quedar instalada es la 1.2.4

Verificar la version de numpy instalada `pip list | grep numpy`


### 🧪 Verificacion de instalacion correcta
```
import numpy as np
import mmcv
import mmdet
import mmpose
import torch

print("NumPy:", np.__version__)    # Debe ser 1.24.4
print("MMCV:", mmcv.__version__)   # Debe estar entre 2.0.0rc4 y 2.1.9
print("MMDet:", mmdet.__version__)
print("MMPose:", mmpose.__version__)
print("PyTorch CPU disponible:", not torch.cuda.is_available())

# Si las cuatro librerías se importan y el NumPy es 1.24.4, ¡la instalación es exitosa!
```

### ❎ Error comun con la libreria mmcv.__ext
Aunque se haya instalado la libreria mmcv-lite (precompilada), existen paquetes dentro de la configuracion que mandan a llamar a la libreria completa (mmcv), por lo que brindara un error, para solucionar el problema, hay que deshabilitar esas importaciones de forma manual, para ello, es necesario entrar a los paquetes instalados dentro del entorno virtual en una ruta como la siguiente
```
[RUTA_BASE]/MMPosev1.1/mmpose_env/lib/python3.10/site-packages/mmpose/models/heads/

# Ejemplo real
/Users/apple/Documents/uziel/MMPosev1.1/mmpose_env/lib/python3.10/site-packages/mmpose/models/heads/
```

Deshabilitar las siguietnes lineas
```
# ... otras importaciones
# from .transformer_heads import EDPoseHead # Comentar o borrar esta línea

__all__ = [
    # ... otros heads
    # 'EDPoseHead', # Comentar o borrar esta línea
]
```