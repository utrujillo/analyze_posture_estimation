# Configuraciones necesarias para la instalacion de ViTPose

### 🚀 Instalacion de ViTPose utilizando Conda

> La instalacion parte de que ya se tiene instalado Anaconda en el computador, puede descargarse del siguiente sitio web <a href="https://www.anaconda.com/download/success">Conda</a> 

La documentacion de ViTPose puede ser encontrada en su sitio web <a href="https://huggingface.co/docs/transformers/main/en/model_doc/vitpose#vitpose">ViTPose Docs</a>


**1. Creacion de entorno virutal**
```
conda create -p ./vitpose_env python=3.11 -y
conda activate ./vitpose_env
conda install pip p -y
```

**2. Instalacion de dependencias estandar**
```
pip install requests supervision transformers
pip install numpy==1.26.4 --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-deps
pip install accelerate
```

**3. Instalar otros paquetes necesarios para las pruebas**
```
pip install pandas openpyxl seaborn python-dotenv tqdm ipykernel
```

> Es importante instalar pandas e ipynerkel al final ya que la compilacion de las librerias de mmcv-lite requeiren la version de numpy 1.2.4, pero al instalr pandas, esta version de numpy se actualiza a una mas reciente ej. 2.2.6

**4. Reinstalar numpy a la version que ocupa mmpose**
```
pip uninstall numpy -y
pip install numpy==1.24.4
```
> Al finalizar todas las instalaciones, la version de numpy que debe quedar instalada es la 1.2.4

Verificar la version de numpy instalada `pip list | grep numpy`

**5. Ejecuta el script test.py para probar la inferencia**
```
python test.py
```