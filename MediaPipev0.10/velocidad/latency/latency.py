import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

class Latency:
    def __init__(self, basepath, images_path, labels_path, videos_path):
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        self.keypoint_mapping = {
            0: self.mp_pose.PoseLandmark.NOSE,           # Nose
            1: self.mp_pose.PoseLandmark.LEFT_EYE,       # Left-eye
            2: self.mp_pose.PoseLandmark.RIGHT_EYE,      # Right-eye
            3: self.mp_pose.PoseLandmark.LEFT_EAR,       # Left-ear
            4: self.mp_pose.PoseLandmark.RIGHT_EAR,      # Right-ear
            5: self.mp_pose.PoseLandmark.LEFT_SHOULDER,  # Left-shoulder
            6: self.mp_pose.PoseLandmark.RIGHT_SHOULDER, # Right-shoulder
            7: self.mp_pose.PoseLandmark.LEFT_ELBOW,     # Left-elbow
            8: self.mp_pose.PoseLandmark.RIGHT_ELBOW,    # Right-elbow
            9: self.mp_pose.PoseLandmark.LEFT_WRIST,     # Left-wrist
            10: self.mp_pose.PoseLandmark.RIGHT_WRIST,   # Right-wrist
            11: self.mp_pose.PoseLandmark.LEFT_HIP,      # Left-hip
            12: self.mp_pose.PoseLandmark.RIGHT_HIP,     # Right-hip
            13: self.mp_pose.PoseLandmark.LEFT_KNEE,     # Left-knee
            14: self.mp_pose.PoseLandmark.RIGHT_KNEE,    # Right-knee
            15: self.mp_pose.PoseLandmark.LEFT_ANKLE,    # Left-ankle
            16: self.mp_pose.PoseLandmark.RIGHT_ANKLE    # Right-ankle
        }
    
    def __file_exists(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    def evaluate_image(self, image, threshold, results):
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        # Process the image
        results_pose = self.pose.process(img_rgb)
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
        Evalúa la latencia de inferencia para un video grabado (archivo)
        
        Args:
            video_path (str): Ruta completa al archivo de video
            threshold (float): Umbral de confianza para las detecciones
            results (list): Lista para almacenar los resultados
            frame_skip (int): Procesar 1 de cada N frames (default: 1)
            max_frames (int): Límite de frames a procesar (opcional)
            
        Returns:
            list: Resultados actualizados con métricas del video
        """
        video = os.path.join(self.videos_path, videos_path)
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
        for _ in tqdm(range(total_frames), desc=f"Procesando {os.path.basename(video)}"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Saltar frames según frame_skip
            if processed_frames % frame_skip != 0:
                processed_frames += 1
                continue
                
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Medir latencia por frame
            start_time = time.time()
            _ = self.pose.process(frame_rgb)
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

    def __del__(self):
        # Close MediaPipe Pose
        if hasattr(self, 'pose'):
            self.pose.close()