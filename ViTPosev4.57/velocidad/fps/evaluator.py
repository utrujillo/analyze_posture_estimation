import os
import cv2
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class Evaluator:
    def __init__(self, basepath, images_path, labels_path, videos_path, detector_path, estimator_path):
        """
        Inicializa el evaluador con ViTPose (Transformers).
        
        :param basepath: Ruta base del proyecto
        :param images_path: Ruta de las imágenes
        :param labels_path: Ruta de las etiquetas
        :param videos_path: Ruta de los videos
        :param detector_path: Ruta local del modelo RTDetr para detección de personas.
        :param estimator_path: Ruta local del modelo ViTPose para estimación de pose.
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        # Inicializar el dispositivo por defecto (se moverá más tarde)
        self.device = torch.device('cpu') 
        
        print("Cargando modelos ViTPose y RTDetr...")

        # --- 1. Inicializar el Detector de Personas (RT-DETR) ---
        self.__file_exists(detector_path)
        self.person_image_processor = AutoProcessor.from_pretrained(detector_path, use_fast=True)
        # Cargamos en CPU primero para facilitar el manejo de dispositivos
        self.person_model = RTDetrForObjectDetection.from_pretrained(detector_path, device_map='cpu') 
        self.person_model.eval()

        # --- 2. Inicializar el Estimador de Pose (ViTPose) ---
        self.__file_exists(estimator_path)
        self.pose_image_processor = AutoProcessor.from_pretrained(estimator_path, use_fast=True)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(estimator_path, device_map='cpu')
        self.pose_model.eval()
    
    def __file_exists(self, file_path):
        """Verifica si un archivo o directorio existe."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo/directorio '{file_path}' no existe.")

    def __get_target_device(self, device_type):
        """Determina el dispositivo target (cpu, cuda, mps)."""
        if device_type == 'gpu':
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available(): # Soporte para Apple Silicon
                return torch.device("mps")
            raise RuntimeError("GPU/MPS solicitado pero no disponible.")
        return torch.device("cpu")
    
    def __move_models_to_device(self, device):
        """Mueve ambos modelos (RTDetr y ViTPose) al dispositivo especificado."""
        # Mover RTDetr
        current_det_device = next(self.person_model.parameters()).device
        if current_det_device != device:
            self.person_model.to(device)

        # Mover ViTPose
        current_pose_device = next(self.pose_model.parameters()).device
        if current_pose_device != device:
            self.pose_model.to(device)
        
        self.device = device # Actualizar el dispositivo de la clase
            
    def __run_two_step_inference(self, frame, device):
        """
        Ejecuta la inferencia de dos pasos (RTDetr + ViTPose).
        
        Args:
            frame (np.ndarray): Imagen/frame en formato BGR de OpenCV.
            
        Returns:
            int: Número de personas cuya pose fue estimada.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        height, width = frame.shape[:2]

        # --- 1. Detección de Personas (RT-DETR) ---
        inputs_det = self.person_image_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_det = self.person_model(**inputs_det)

        results_det = self.person_image_processor.post_process_object_detection(
            outputs_det, target_sizes=torch.tensor([(height, width)]), threshold=0.5
        )
        person_boxes = results_det[0]["boxes"][results_det[0]["labels"] == 0].cpu().numpy()

        # Convertir cajas de VOC a COCO
        person_boxes_coco = person_boxes.copy()
        person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]
        person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]
        
        if len(person_boxes_coco) == 0:
            return 0, None # Retorna 0 detecciones y None como keypoints

        # --- 2. Estimación de Pose (ViTPose) ---
        inputs_pose = self.pose_image_processor(pil_image, boxes=[person_boxes_coco.tolist()], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=[person_boxes_coco.tolist()])
        
        # Contar instancias y extraer keypoints de la primera persona para el log
        num_keypoints_sets = len(pose_results[0]) if pose_results[0] else 0
        keypoints_data = pose_results[0][0]['keypoints'].cpu().numpy() if num_keypoints_sets > 0 else None

        return num_keypoints_sets, keypoints_data

    # --------------------------------------------------------------------------
    # MÉTODOS DE INFERENCIA DE IMAGEN
    # --------------------------------------------------------------------------

    def on_cpu(self, image, threshold, results):
        """Ejecuta inferencia en CPU para una imagen."""
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)
        
        device = self.__get_target_device('cpu')
        self.__move_models_to_device(device)
        
        img = cv2.imread(image_path)
            
        start_time = time.time()
        
        # Inferencia con ViTPose
        keypoints_detected, _ = self.__run_two_step_inference(img, device)
        
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        
        # Obtener dimensiones de la imagen
        image_size = f"{img.shape[1]}x{img.shape[0]}" if img is not None else 'N/A'
        
        results.append({
            'tipo': str(device),
            'image': image,
            'image_size': image_size,
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps,
            'keypoints_detected': keypoints_detected
        })

        return results

    def on_gpu(self, image, threshold, results):
        """Ejecuta inferencia en GPU/MPS para una imagen."""
        
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)
        
        try:
            device = self.__get_target_device('gpu')
            self.__move_models_to_device(device)
        except RuntimeError as e:
            print(f"Advertencia: {e}")
            return results # Si la GPU no está disponible, salimos.
        
        img = cv2.imread(image_path)
        
        start_time = time.time()
        
        # Inferencia con ViTPose
        keypoints_detected, _ = self.__run_two_step_inference(img, device)
        
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        
        # Obtener dimensiones de la imagen
        image_size = f"{img.shape[1]}x{img.shape[0]}" if img is not None else 'N/A'
        
        results.append({
            'tipo': str(device),
            'image': image,
            'image_size': image_size,
            'threshold': threshold,
            'time': elapsed_time,
            'fps': fps,
            'keypoints_detected': keypoints_detected
        })

        return results
    
    # --------------------------------------------------------------------------
    # MÉTODOS DE INFERENCIA DE VIDEO
    # --------------------------------------------------------------------------

    def video_on_cpu(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        """Ejecuta inferencia en CPU para un video, mostrando el progreso."""
        
        video = videos_path # Asumimos que videos_path es la ruta completa
        self.__file_exists(video)
        
        device = self.__get_target_device('cpu')
        self.__move_models_to_device(device)
            
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        video_name = os.path.basename(videos_path)
        total_frames_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Cálculo de frames a procesar para tqdm
        total_frames_to_process = total_frames_file // frame_skip
        if max_frames is not None:
            total_frames_to_process = min(total_frames_to_process, max_frames)
            
        processed_frames_count = 0
        frame_counter = 0 # Contador de frames leídos del archivo
        
        # Procesar frames con barra de progreso
        pbar = tqdm(total=total_frames_to_process, desc=f"CPU: Procesando ({video_name}) (th:{threshold})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_counter % frame_skip == 0:
                
                if max_frames is not None and processed_frames_count >= max_frames:
                    break
                    
                start_time = time.time()
                
                # Inferencia con ViTPose
                keypoints_detected, _ = self.__run_two_step_inference(frame, device)
                
                elapsed_time = time.time() - start_time
                fps_inference = 1 / elapsed_time if elapsed_time > 0 else float('inf')
                processed_frames_count += 1
                
                results.append({
                    'tipo': str(device),
                    'video': video_name,
                    'frame': frame_counter,
                    'video_fps': fps_video, 
                    'frame_skip': frame_skip, 
                    'image_size': f"{width}x{height}",
                    'threshold': threshold,
                    'time': elapsed_time,
                    'fps': fps_inference,
                    'keypoints_detected': keypoints_detected
                })
                pbar.update(1)
            
            frame_counter += 1
        
        cap.release()
        pbar.close()
        return results

    def video_on_gpu(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        """Ejecuta inferencia en GPU/MPS para un video, mostrando el progreso."""
        
        video = videos_path # Asumimos que videos_path es la ruta completa
        self.__file_exists(video)
        
        try:
            device = self.__get_target_device('gpu')
            self.__move_models_to_device(device)
        except RuntimeError as e:
            print(f"Advertencia: {e}")
            return results # Si la GPU no está disponible, salimos.
        
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video}")
        
        video_name = os.path.basename(videos_path)
        total_frames_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Cálculo de frames a procesar para tqdm
        total_frames_to_process = total_frames_file // frame_skip
        if max_frames is not None:
            total_frames_to_process = min(total_frames_to_process, max_frames)
            
        processed_frames_count = 0
        frame_counter = 0 # Contador de frames leídos del archivo
        
        # Procesar frames con barra de progreso
        pbar = tqdm(total=total_frames_to_process, desc=f"GPU: Procesando ({video_name}) (th:{threshold})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_counter % frame_skip == 0:
                
                if max_frames is not None and processed_frames_count >= max_frames:
                    break
                
                start_time = time.time()
                
                # Inferencia con ViTPose
                keypoints_detected, _ = self.__run_two_step_inference(frame, device)
                
                elapsed_time = time.time() - start_time
                fps_inference = 1 / elapsed_time if elapsed_time > 0 else float('inf')
                processed_frames_count += 1
                
                results.append({
                    'tipo': str(device),
                    'video': video_name,
                    'frame': frame_counter,
                    'video_fps': fps_video, 
                    'frame_skip': frame_skip, 
                    'image_size': f"{width}x{height}",
                    'threshold': threshold,
                    'time': elapsed_time,
                    'fps': fps_inference,
                    'keypoints_detected': keypoints_detected
                })
                pbar.update(1)

            frame_counter += 1
        
        cap.release()
        pbar.close()
        return results