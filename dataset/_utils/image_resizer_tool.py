import os
from PIL import Image

def procesar_imagenes(input_dir, output_dir, tamaño_maximo=256, inicio=None):
    """
    Procesa y renombra imágenes de una carpeta de entrada, redimensionándolas
    y guardándolas secuencialmente en la carpeta de salida.

    Parámetros:
    - input_dir: ruta de la carpeta con las imágenes originales
    - output_dir: ruta donde se guardarán las imágenes procesadas
    - tamaño_maximo: tamaño máximo en píxeles (ancho o alto)
    - inicio: número desde el cual comenzar la numeración (opcional)
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Extensiones permitidas
    formatos_validos = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.avif', '.webp')

    # Listar imágenes válidas en la carpeta de entrada
    imagenes = [f for f in os.listdir(input_dir) if f.lower().endswith(formatos_validos)]

    # Obtener fecha de modificación y ordenar por fecha
    imagenes_con_fecha = []
    for nombre_imagen in imagenes:
        ruta_imagen = os.path.join(input_dir, nombre_imagen)
        fecha_creacion = os.path.getmtime(ruta_imagen)
        imagenes_con_fecha.append((nombre_imagen, fecha_creacion))

    imagenes_con_fecha.sort(key=lambda x: x[1])
    imagenes_ordenadas = [x[0] for x in imagenes_con_fecha]

    # Determinar número inicial automático si no se pasa "inicio"
    if inicio is None:
        existentes = [
            f for f in os.listdir(output_dir)
            if f.lower().startswith("imagen_") and f.lower().endswith(".jpg")
        ]
        max_num = 0
        for f in existentes:
            try:
                num = int(f.split("_")[1].split(".")[0])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                continue
        inicio = max_num + 1  # continuar automáticamente

    # Procesar y guardar imágenes
    for i, nombre_imagen in enumerate(imagenes_ordenadas, start=inicio):
        ruta_imagen = os.path.join(input_dir, nombre_imagen)
        img = Image.open(ruta_imagen)

        ancho_original, alto_original = img.size

        # Mantener la proporción
        if alto_original > ancho_original:
            nuevo_alto = tamaño_maximo
            relacion_aspecto = ancho_original / alto_original
            nuevo_ancho = int(tamaño_maximo * relacion_aspecto)
        elif ancho_original > alto_original:
            nuevo_ancho = tamaño_maximo
            relacion_aspecto = alto_original / ancho_original
            nuevo_alto = int(tamaño_maximo * relacion_aspecto)
        else:
            nuevo_ancho = nuevo_alto = tamaño_maximo

        img_redimensionada = img.resize((nuevo_ancho, nuevo_alto))

        if img_redimensionada.mode == "RGBA":
            img_redimensionada = img_redimensionada.convert("RGB")

        nuevo_nombre = f"imagen_{i:03d}.jpg"
        img_redimensionada.save(os.path.join(output_dir, nuevo_nombre), format="JPEG")

    print(f"Procesamiento completado. Imágenes guardadas en: {output_dir}")
    print(f"Comenzó desde imagen_{inicio:03d}.jpg")

# Ejemplo de uso:
input_dir = '/Users/apple/Documents/imagenes'
output_dir = '/Users/apple/Documents/imagenes/out'

# Puedes especificar el número desde el cual continuar:
procesar_imagenes(input_dir, output_dir, tamaño_maximo=256, inicio=73)
