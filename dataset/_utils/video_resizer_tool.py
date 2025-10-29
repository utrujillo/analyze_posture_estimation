import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import cv2 

# Cargar variables de entorno
load_dotenv(Path(__file__).resolve().parents[2] / '.env')


def calculate_new_dimensions(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Calcula las nuevas dimensiones manteniendo la proporción."""
    if max(width, height) <= max_size:
        return width, height

    if width >= height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    # Asegura que las dimensiones sean positivas y pares (necesario para algunos códecs)
    new_width = max(1, new_width // 2 * 2)
    new_height = max(1, new_height // 2 * 2)

    return new_width, new_height


def resize_video_cv2(input_path: Path, output_path: Path, max_size: int):
    """Redimensiona un único video usando cv2 y lo guarda."""
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        tqdm.write(f"⚠️ Error: No se pudo abrir el video {input_path.name}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    new_width, new_height = calculate_new_dimensions(original_width, original_height, max_size)

    if new_width == original_width and new_height == original_height:
        tqdm.write(f"   - Ya tiene el tamaño adecuado. Omitiendo.")
        cap.release()
        return

    tqdm.write(f"   - Redimensionando de {original_width}x{original_height} a {new_width}x{new_height}")

    # Preparar la ruta de salida y crear directorios
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Intenta usar 'mp4v' para .mp4 (amplia compatibilidad con OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # Aseguramos que la extensión de salida sea .mp4
    final_output_path = output_path.with_suffix('.mp4')
    
    out = cv2.VideoWriter(str(final_output_path), fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        # Mensaje de error si el códec no funciona
        tqdm.write("❌ Error fatal: No se pudo configurar VideoWriter. Intenta instalar más códecs (ej. FFmpeg).")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    cap.release()
    out.release()
    tqdm.write(f"   - ✅ Guardado en: {final_output_path.relative_to(Path(os.environ['VIDEO_RESIZE_OUTPUT']))}")


def main():
    # 1. Obtener parámetros desde las variables de entorno
    video_input_dir_str = os.environ.get('VIDEO_RESIZE_INPUT')
    video_output_dir_str = os.environ.get('VIDEO_RESIZE_OUTPUT')
    max_size_str = os.environ.get('MAX_SIZE')

    if not all([video_input_dir_str, video_output_dir_str, max_size_str]):
        print("❌ Error: Asegúrate de que VIDEO_RESIZE_INPUT, VIDEO_RESIZE_OUTPUT y MAX_SIZE estén definidos en el .env")
        sys.exit(1)

    try:
        max_size = int(max_size_str)
    except ValueError:
        print(f"❌ Error: MAX_SIZE debe ser un número entero. Valor actual: {max_size_str}")
        sys.exit(1)

    video_input_dir = Path(video_input_dir_str)
    video_output_dir = Path(video_output_dir_str)

    if not video_input_dir.is_dir():
        print(f"❌ Error: El directorio de entrada no existe: {video_input_dir}")
        sys.exit(1)

    print(f"🎥 Iniciando redimensionamiento de videos (OpenCV/CV2)")
    print("-" * 50)
    print(f"  Directorio de entrada: {video_input_dir}")
    print(f"  Directorio de salida: {video_output_dir}")
    print(f"  Máximo tamaño del lado: {max_size} px\n")

    # 2. Buscar videos de forma recursiva
    video_extensions = ['*.mp4', '*.avi']
    found_videos = []
    for ext in video_extensions:
        found_videos.extend(list(video_input_dir.rglob(ext)))

    if not found_videos:
        print("⚠️ No se encontraron videos .mp4 o .avi para procesar.")
        return

    total_videos = len(found_videos)
    print(f"🔎 {total_videos} videos encontrados para procesar.")
    
    # 3. Iterar sobre los videos con tqdm
    for i, input_video_path in enumerate(tqdm(found_videos, desc="Procesando videos", unit="archivo")):
        
        # Calcular la ruta relativa para replicar la estructura
        relative_path = input_video_path.relative_to(video_input_dir)
        
        # Construir la ruta de salida
        output_video_path = video_output_dir / relative_path

        tqdm.write(f"\n-> Archivo {i+1}/{total_videos}: {input_video_path.relative_to(video_input_dir)}")
        
        resize_video_cv2(input_video_path, output_video_path, max_size)

    print("-" * 50)
    print("✅ Proceso de redimensionamiento de videos finalizado.")


if __name__ == "__main__":
    main()