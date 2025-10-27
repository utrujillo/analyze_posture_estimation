import os, cv2, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

class Latency:
    def __init__(self, basepath, images_path, labels_path, videos_path,model_name='yolo11n-pose.pt'):
        self.basepath = basepath
        self.model_path = os.path.join(basepath, model_name)
        self.model = YOLO(self.model_path)
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
    
    def __file_exists(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"La archivo '{file_path}' no existe.")
    
    def evaluate_image(self, image, threshold, results):
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)


        start_time = time.time()
        inference = self.model(image_path)
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        results.append({
            'image': image,
            'image_size': f"{width}x{height}",
            'threshold': threshold,
            'latency': latency_ms
        })

        return results
    
    def evaluate_video(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        """
        Evalúa la latencia de inferencia para un video grabado (archivo) usando ViTPose.
        
        Args:
            videos_path (str): Ruta completa al archivo de video.
            threshold (float): Umbral de confianza para las detecciones.
            results (list): Lista para almacenar los resultados.
            frame_skip (int): Procesar 1 de cada N frames (default: 1).
            max_frames (int): Límite de frames a procesar (opcional).
            
        Returns:
            list: Resultados actualizados con métricas del video.
        """
        video = os.path.join(self.videos_path, videos_path) # Usamos self.videos_path
        self.__file_exists(video)
        
        # Abrir el video
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        # Obtener propiedades del video
        total_frames_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calcular el número total de frames a procesar (para tqdm)
        # 1. Total de frames disponibles para ser procesados (respetando frame_skip)
        frames_available_to_process = total_frames_file // frame_skip
        # 2. Aplicar el límite de max_frames
        total_frames_to_process = frames_available_to_process
        if max_frames is not None:
            total_frames_to_process = min(frames_available_to_process, max_frames)
        
        # Variables para métricas
        frame_latencies = []
        frame_counter = 0 # Contador de frames leídos del archivo
        
        # Inicializar la barra de progreso
        pbar = tqdm(total=total_frames_to_process, desc=f"Procesando {videos_path} (th:{threshold})")

        while True:
            ret, frame = cap.read()
            if not ret:
                # El video terminó
                break
                
            # Verificar si se debe procesar este frame (salto)
            if frame_counter % frame_skip == 0:
                
                # Limitar el número total de frames procesados (max_frames)
                if max_frames is not None and len(frame_latencies) >= max_frames:
                    break
                
                # --- Lógica de Medición de Latencia ---
                
                # Aquí simulamos la inferencia de dos pasos (det + pose)
                # En tu script original usabas self.model(frame, conf=threshold)
                # Dado que la estructura original no tiene __run_two_step_inference,
                # usaré self.model(frame) y ajustaré la estructura del resultado para tu solicitud.
                
                start_time = time.time()
                # La inferencia YOLOv8/v11 Pose integra detección y pose.
                # Usamos el conf=threshold para control de detección.
                _ = self.model(frame, conf=threshold, verbose=False) 
                latency_ms = (time.time() - start_time) * 1000
                frame_latencies.append(latency_ms)
                
                # Agregar resultados
                results.append({
                    'video': os.path.basename(videos_path),
                    'video_resolution': f"{width}x{height}",
                    'threshold': threshold,
                    'video_fps': fps,
                    'frame_skip': frame_skip,
                    'latency_ms': latency_ms,
                    'processed_frames': len(frame_latencies),
                    'total_frames_in_file': total_frames_file,
                })
                
                # Actualizar la barra de progreso por cada frame procesado
                pbar.update(1)

            frame_counter += 1
        
        pbar.close()
        cap.release()
        return results