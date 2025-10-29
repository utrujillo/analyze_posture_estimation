import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from typing import List, Dict, Union, Optional

class Evaluator:
    def __init__(self, basepath, images_path, labels_path, videos_path):
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Keypoint mapping
        self.keypoint_mapping = {
            0: self.mp_pose.PoseLandmark.NOSE,
            1: self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
            2: self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
            3: self.mp_pose.PoseLandmark.LEFT_EAR,
            4: self.mp_pose.PoseLandmark.RIGHT_EAR,
            5: self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            6: self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            7: self.mp_pose.PoseLandmark.LEFT_ELBOW,
            8: self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            9: self.mp_pose.PoseLandmark.LEFT_WRIST,
            10: self.mp_pose.PoseLandmark.RIGHT_WRIST,
            11: self.mp_pose.PoseLandmark.LEFT_HIP,
            12: self.mp_pose.PoseLandmark.RIGHT_HIP,
            13: self.mp_pose.PoseLandmark.LEFT_KNEE,
            14: self.mp_pose.PoseLandmark.RIGHT_KNEE,
            15: self.mp_pose.PoseLandmark.LEFT_ANKLE,
            16: self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }
        
        # Initialize CPU and GPU instances
        self.cpu_pose = None
        self.gpu_pose = None
    
    def __file_exists(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    def _init_cpu_instance(self):
        """Initialize CPU instance of MediaPipe Pose"""
        if self.cpu_pose is None:
            self.cpu_pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
    
    def _init_gpu_instance(self):
        """Initialize GPU instance of MediaPipe Pose"""
        if self.gpu_pose is None:
            self.gpu_pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            # MediaPipe automatically uses GPU when available through OpenGL/Vulkan
    
    def process_image(self, image_path: str, threshold: float, use_gpu: bool = False) -> Dict:
        """Process a single image and return detection results."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Select the appropriate pose instance
        pose_instance = self.gpu_pose if use_gpu else self.cpu_pose
        
        start_time = time.time()
        results = pose_instance.process(image_rgb)
        elapsed_time = time.time() - start_time
        
        fps = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
        
        return {
            'image_size': f"{width}x{height}",
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps,
            'landmarks': results.pose_landmarks
        }
    
    def on_cpu(self, image: str, threshold: float, results: List[Dict]) -> List[Dict]:
        """Process image on CPU and append results."""
        self._init_cpu_instance()
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)
        
        processed = self.process_image(image_path, threshold, use_gpu=False)
        
        results.append({
            'tipo': 'CPU',
            'image': image,
            'image_size': processed['image_size'],
            'threshold': threshold,
            'time': processed['time'],
            'fps': processed['fps']
        })
        
        return results
    
    def on_gpu(self, image: str, threshold: float, results: List[Dict]) -> List[Dict]:
        """Process image on GPU and append results."""
        self._init_gpu_instance()
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)
        
        processed = self.process_image(image_path, threshold, use_gpu=True)
        
        results.append({
            'tipo': 'GPU',
            'image': image,
            'image_size': processed['image_size'],
            'threshold': threshold,
            'time': processed['time'],
            'fps': processed['fps']
        })
        
        return results
    
    def process_video_frame(self, frame: np.ndarray, threshold: float, use_gpu: bool = False) -> Dict:
        """Process a single video frame and return detection results."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        
        # Select the appropriate pose instance
        pose_instance = self.gpu_pose if use_gpu else self.cpu_pose
        
        start_time = time.time()
        results = pose_instance.process(frame_rgb)
        elapsed_time = time.time() - start_time
        
        fps = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
        
        return {
            'image_size': f"{width}x{height}",
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps,
            'landmarks': results.pose_landmarks
        }
    
    def video_on_cpu(self, videos_path: str, threshold: float, results: List[Dict], 
                    frame_skip: int = 1, max_frames: Optional[int] = None) -> List[Dict]:
        """Process video on CPU and append results with progress bar."""
        self._init_cpu_instance()
        video = os.path.join(self.videos_path, videos_path)
        self.__file_exists(video)
        
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1. Calcular el número total de frames que el bucle va a intentar leer.
        if max_frames is not None:
            total_frames = min(total_frames, max_frames * frame_skip)

        # 2. Calcular el número REAL de frames que se van a procesar.
        #    Esto es 'total_frames' dividido por 'frame_skip' (redondeado hacia arriba).
        #    Usamos el total de frames leídos para el rango de iteración:
        
        video_basename = os.path.basename(videos_path)
        
        # 3. Envolver el rango con tqdm y usar 'total=total_frames'
        for i in tqdm(range(total_frames), desc=f"CPU: {video_basename}", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                # La barra de progreso terminará cuando termine el bucle
                break
            
            # Solo procesar cada 'frame_skip' frames
            if i % frame_skip == 0:
                # Nota: He quitado el print interno para que no interfiera con tqdm
                processed = self.process_video_frame(frame, threshold, use_gpu=False)
                
                results.append({
                    'tipo': 'CPU',
                    'video': video_basename,
                    'frame': i,
                    'image_size': processed['image_size'],
                    'threshold': threshold,
                    'time': processed['time'],
                    'fps': processed['fps']
                })
        
        cap.release()
        return results
    
    def video_on_gpu(self, videos_path: str, threshold: float, results: List[Dict], 
                    frame_skip: int = 1, max_frames: Optional[int] = None) -> List[Dict]:
        """Process video on GPU and append results with progress bar."""
        self._init_gpu_instance()
        video = os.path.join(self.videos_path, videos_path)
        self.__file_exists(video)
        
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1. Calcular el número total de frames que el bucle va a intentar leer.
        if max_frames is not None:
            total_frames = min(total_frames, max_frames * frame_skip)

        video_basename = os.path.basename(videos_path)

        # 2. Envolver el rango con tqdm y usar 'total=total_frames'
        for i in tqdm(range(total_frames), desc=f"GPU: {video_basename}", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Solo procesar cada 'frame_skip' frames
            if i % frame_skip == 0:
                # Nota: He quitado el print interno para que no interfiera con tqdm
                processed = self.process_video_frame(frame, threshold, use_gpu=True)
                
                results.append({
                    'tipo': 'GPU',
                    'video': video_basename,
                    'frame': i,
                    'image_size': processed['image_size'],
                    'threshold': threshold,
                    'time': processed['time'],
                    'fps': processed['fps']
                })
        
        cap.release()
        return results
    
    def close(self):
        """Close MediaPipe Pose resources."""
        if self.cpu_pose:
            self.cpu_pose.close()
        if self.gpu_pose:
            self.gpu_pose.close()