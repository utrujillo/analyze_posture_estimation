import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample

class Latency:
    def __init__(self, basepath, images_path, labels_path, videos_path, config_file, checkpoint_file):
        """
        Inicializa el evaluador de latencia con MMPose.
        
        :param basepath: Ruta base del proyecto
        :param images_path: Ruta de las imágenes
        :param labels_path: Ruta de las etiquetas
        :param config_file: Archivo de configuración del modelo MMPose
        :param checkpoint_file: Checkpoint del modelo MMPose
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        
        # Construir rutas completas para los archivos de configuración y checkpoint
        config_path = os.path.join(basepath, config_file)
        checkpoint_path = os.path.join(basepath, checkpoint_file)
        
        # Verificar que los archivos existan
        self.__file_exists(config_path)
        self.__file_exists(checkpoint_path)
        
        # Inicializar el modelo MMPose (HRNet en este caso)
        self.model = init_model(config_path, checkpoint_path, device='cpu')
    
    def __file_exists(self, file_path):
        """Verifica si un archivo existe."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    def evaluate_image(self, image, threshold, results):
        """
        Evalúa la latencia de inferencia para una imagen.
        
        Args:
            image (str): Nombre del archivo de imagen
            threshold (float): Umbral de confianza para los keypoints
            results (list): Lista para almacenar los resultados
            
        Returns:
            list: Resultados actualizados con métricas de la imagen
        """
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)

        # Leer imagen y obtener dimensiones
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Medir latencia con inference_topdown
        start_time = time.time()
        _ = inference_topdown(self.model, img)  # Inferencia con MMPose
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Almacenar resultados
        results.append({
            'image': image,
            'image_size': f"{width}x{height}",
            'threshold': threshold,
            'latency': latency_ms
        })

        return results
    
    def evaluate_video(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        """
        Evalúa la latencia de inferencia para un video grabado (archivo)
        
        Args:
            video_path (str): Nombre del archivo de video
            threshold (float): Umbral de confianza para los keypoints
            results (list): Lista para almacenar los resultados
            frame_skip (int): Procesar 1 de cada N frames (default: 1)
            max_frames (int): Límite de frames a procesar (opcional)
            
        Returns:
            list: Resultados actualizados con métricas del video
        """
        video = os.path.join(self.basepath, videos_path)
        self.__file_exists(video)
        
        # Abrir el video
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        # Obtener propiedades del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames is not None:
            total_frames = min(total_frames, max_frames * frame_skip)
        
        # Variables para métricas
        frame_latencies = []
        processed_frames = 0
        
        # Procesar frames con barra de progreso
        for _ in tqdm(range(total_frames), desc=f"Procesando ({videos_path}) - (th:{threshold})"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Saltar frames según frame_skip
            if processed_frames % frame_skip != 0:
                processed_frames += 1
                continue
                
            # Medir latencia por frame con inference_topdown
            start_time = time.time()
            _ = inference_topdown(self.model, frame)  # Inferencia con MMPose
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
                'total_frames': total_frames,
            })

            processed_frames += 1
        
        cap.release()
        return results