import os, cv2, torch, time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm # ¡IMPORTANTE: Añadir tqdm!

class Evaluator:
    def __init__(self, basepath, images_path, labels_path, videos_path, model_name='yolo11n-pose.pt'):
        self.basepath = basepath
        self.model_path = os.path.join(basepath, model_name)
        self.model = YOLO(self.model_path)
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
    
    def __file_exists(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"La archivo '{file_path}' no existe.")
    
    def on_cpu(self, image, threshold, results):
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)
        
        device = torch.device('cpu')
        start_time = time.time()
        results_pred = self.model(source=image_path, device=device, conf=threshold)
        image_size = results_pred[0].orig_shape if results_pred else 'N/A'
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
        results.append({
            'tipo': device,
            'image': image,
            'image_size': image_size,
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps
        })

        return results

    def on_gpu(self, image, threshold, results):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS (GPU) no disponible en este sistema.")
        device = torch.device("mps")
        image_path = os.path.join(self.images_path, image)
        
        # Corrección: self.__file_exists solo acepta un argumento
        self.__file_exists(image_path) 
        
        start_time = time.time()
        results_pred = self.model(source=image_path, device=device, conf=threshold)
        image_size = results_pred[0].orig_shape if results_pred else 'N/A'
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
        results.append({
            'tipo': device,
            'image': image,
            'image_size': image_size,
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps
        })

        return results
    
    # ----------------------------------------------------------------------
    # MÉTODO MODIFICADO: video_on_cpu con tqdm
    # ----------------------------------------------------------------------
    def video_on_cpu(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        video = os.path.join(self.videos_path, videos_path)
        self.__file_exists(video)
        
        device = torch.device('cpu')
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        # 1. Obtener propiedades
        total_frames_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 2. Calcular el total de frames a procesar para la barra
        frames_available_to_process = total_frames_file // frame_skip
        total_to_process = frames_available_to_process
        if max_frames is not None:
            total_to_process = min(frames_available_to_process, max_frames)
        
        # 3. Inicializar la barra de progreso
        pbar = tqdm(total=total_to_process, desc=f"Video: {videos_path} (th: {threshold}) - CPU")
        
        frame_counter = 0 # Contador de frames leídos
        processed_count = 0 # Contador de frames procesados
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_counter % frame_skip == 0:
                # Verificar límite máximo de frames procesados
                if max_frames is not None and processed_count >= max_frames:
                    break
                
                # Ejecutar inferencia
                start_time = time.time()
                _ = self.model(source=frame, device=device, conf=threshold, verbose=False)
                elapsed_time = time.time() - start_time
                fps_inf = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
                
                results.append({
                    'tipo': str(device),
                    'video': os.path.basename(videos_path),
                    'frame': frame_counter,
                    'image_size': f"{width}x{height}",
                    'threshold': threshold,
                    'time': elapsed_time,
                    'fps': fps_inf
                })
                
                # Actualizar barra y contador
                pbar.update(1)
                processed_count += 1
            
            frame_counter += 1
        
        pbar.close()
        cap.release()
        return results

    # ----------------------------------------------------------------------
    # MÉTODO MODIFICADO: video_on_gpu con tqdm
    # ----------------------------------------------------------------------
    def video_on_gpu(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS (GPU) no disponible en este sistema.")
            
        device = torch.device("mps")
        video = os.path.join(self.videos_path, videos_path)
        self.__file_exists(video)
        
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
            
        # 1. Obtener propiedades
        total_frames_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 2. Calcular el total de frames a procesar para la barra
        frames_available_to_process = total_frames_file // frame_skip
        total_to_process = frames_available_to_process
        if max_frames is not None:
            total_to_process = min(frames_available_to_process, max_frames)
            
        # 3. Inicializar la barra de progreso
        pbar = tqdm(total=total_to_process, desc=f"Video: {os.path.basename(videos_path)} - GPU (MPS)")
        
        frame_counter = 0 # Contador de frames leídos
        processed_count = 0 # Contador de frames procesados
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_counter % frame_skip == 0: 
                # Verificar límite máximo de frames procesados
                if max_frames is not None and processed_count >= max_frames:
                    break
                    
                # Ejecutar inferencia
                start_time = time.time()
                # Pasar 'frame' directamente a source, ya que es un numpy array
                _ = self.model(source=frame, device=device, conf=threshold, verbose=False)
                elapsed_time = time.time() - start_time
                fps_inf = 1 / elapsed_time if elapsed_time > 0 else 'N/A'
                
                results.append({
                    'tipo': str(device),
                    'video': os.path.basename(videos_path),
                    'frame': frame_counter,
                    'image_size': f"{width}x{height}",
                    'threshold': threshold,
                    'time': elapsed_time,
                    'fps': fps_inf
                })

                # Actualizar barra y contador
                pbar.update(1)
                processed_count += 1
                
            frame_counter += 1
            
        pbar.close()
        cap.release()
        return results