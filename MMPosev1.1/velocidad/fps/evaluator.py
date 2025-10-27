import os
import cv2
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm # Importamos tqdm para la barra de progreso
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

class Evaluator:
    def __init__(self, basepath, images_path, labels_path, videos_path, config_file, checkpoint_file):
        """
        Inicializa el evaluador PCK/Latencia con MMPose.
        
        :param basepath: Ruta base del proyecto
        :param images_path: Ruta de las imágenes
        :param labels_path: Ruta de las etiquetas
        :param videos_path: Ruta de los videos
        :param config_file: Archivo de configuración del modelo MMPose
        :param checkpoint_file: Checkpoint del modelo MMPose
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        
        # Inicializar el modelo MMPose
        # Nota: Asumiendo que config_file y checkpoint_file son rutas relativas 
        # o ya se gestionan fuera de la clase. Usaremos las rutas tal como llegan.
        self.model = init_model(config_file, checkpoint_file, device='cpu')
    
    def __file_exists(self, file_path):
        """Verifica si un archivo existe."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    def on_cpu(self, image, threshold, results):
        """Ejecuta inferencia en CPU para una imagen."""
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)
        
        device = torch.device('cpu')
        
        # Mover modelo a CPU si no está allí (importante si la última llamada fue a GPU)
        if next(self.model.parameters()).device.type != 'cpu':
            self.model = self.model.to('cpu')
            
        start_time = time.time()
        
        # Inferencia con MMPose
        pose_results = inference_topdown(self.model, image_path)
        data_samples = merge_data_samples(pose_results)
        
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
        
        # Obtener dimensiones de la imagen
        img = cv2.imread(image_path)
        image_size = img.shape[:2] if img is not None else 'N/A'
        
        results.append({
            'tipo': str(device),
            'image': image,
            'image_size': image_size,
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps
        })

        return results

    def on_gpu(self, image, threshold, results):
        """Ejecuta inferencia en GPU para una imagen."""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU no disponible en este sistema.")
        
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)
        
        # Mover modelo a GPU si no está ya allí
        if next(self.model.parameters()).device.type != 'cuda':
            self.model = self.model.to('cuda')
        
        device = torch.device('cuda')
        start_time = time.time()
        
        # Inferencia con MMPose
        pose_results = inference_topdown(self.model, image_path)
        data_samples = merge_data_samples(pose_results)
        
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
        
        # Obtener dimensiones de la imagen
        img = cv2.imread(image_path)
        image_size = img.shape[:2] if img is not None else 'N/A'
        
        results.append({
            'tipo': str(device),
            'image': image,
            'image_size': image_size,
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps
        })

        return results
    
    def video_on_cpu(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        """Ejecuta inferencia en CPU para un video, mostrando el progreso."""
        video = os.path.join(self.videos_path, videos_path)
        self.__file_exists(video)
        
        device = torch.device('cpu')
        
        # Mover modelo a CPU
        if next(self.model.parameters()).device.type != 'cpu':
            self.model = self.model.to('cpu')
            
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        video_name = os.path.basename(videos_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames is not None:
            # Calculamos el número total de frames que leeremos
            total_frames_to_read = min(total_frames, max_frames * frame_skip)
        else:
            total_frames_to_read = total_frames
            
        processed_frames = 0
        
        # Procesar frames con barra de progreso
        for i in tqdm(range(total_frames_to_read), desc=f"CPU: Procesando ({videos_path}) (th:{threshold})"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Solo procesamos si cumple con el frame_skip
            if i % frame_skip == 0:
                start_time = time.time()
                
                # Inferencia con MMPose
                pose_results = inference_topdown(self.model, frame)
                data_samples = merge_data_samples(pose_results)
                
                elapsed_time = time.time() - start_time
                fps_inference = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
                processed_frames += 1 # Contar solo los frames procesados
                
                results.append({
                    'tipo': str(device),
                    'video': video_name,
                    'frame': i,
                    'video_fps': fps_video, 
                    'frame_skip': frame_skip, 
                    'image_size': f"{width}x{height}",
                    'threshold': threshold,
                    'time': elapsed_time,
                    'fps': fps_inference,
                    'keypoints': data_samples.pred_instances.keypoints if hasattr(data_samples, 'pred_instances') else None
                })
            
            # Limitador de frames a procesar si se usa max_frames
            if max_frames is not None and processed_frames >= max_frames:
                break
        
        cap.release()
        return results

    def video_on_gpu(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        """Ejecuta inferencia en GPU para un video, mostrando el progreso."""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU no disponible en este sistema.")
        
        video = os.path.join(self.videos_path, videos_path)
        self.__file_exists(video)
        
        # Mover modelo a GPU
        if next(self.model.parameters()).device.type != 'cuda':
            self.model = self.model.to('cuda')
        
        device = torch.device('cuda')
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        video_name = os.path.basename(videos_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames is not None:
            # Calculamos el número total de frames que leeremos
            total_frames_to_read = min(total_frames, max_frames * frame_skip)
        else:
            total_frames_to_read = total_frames
            
        processed_frames = 0
        
        # Procesar frames con barra de progreso
        for i in tqdm(range(total_frames_to_read), desc=f"GPU: Procesando ({videos_path}) (th:{threshold})"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Solo procesamos si cumple con el frame_skip
            if i % frame_skip == 0:
                start_time = time.time()
                
                # Inferencia con MMPose
                pose_results = inference_topdown(self.model, frame)
                data_samples = merge_data_samples(pose_results)
                
                elapsed_time = time.time() - start_time
                fps_inference = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
                processed_frames += 1 # Contar solo los frames procesados
                
                results.append({
                    'tipo': str(device),
                    'video': video_name,
                    'frame': i,
                    'video_fps': fps_video, 
                    'frame_skip': frame_skip, 
                    'image_size': f"{width}x{height}",
                    'threshold': threshold,
                    'time': elapsed_time,
                    'fps': fps_inference,
                    'keypoints': data_samples.pred_instances.keypoints if hasattr(data_samples, 'pred_instances') else None
                })

            # Limitador de frames a procesar si se usa max_frames
            if max_frames is not None and processed_frames >= max_frames:
                break
        
        cap.release()
        return results