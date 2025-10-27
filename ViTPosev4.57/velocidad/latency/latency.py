import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
# Reemplazamos MMPose y usamos Hugging Face Transformers
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class Latency:
    # NOTA: Los argumentos config_file y checkpoint_file deben ser reemplazados
    # por detector_path y estimator_path en su script de inicialización.
    def __init__(self, basepath, images_path, labels_path, videos_path, detector_path, estimator_path):
        """
        Inicializa el evaluador de latencia con ViTPose (Transformers).
        
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Cargando modelos en el dispositivo: {self.device}")

        # --- 1. Inicializar el Detector de Personas (RT-DETR) ---
        self.__file_exists(detector_path)
        self.person_image_processor = AutoProcessor.from_pretrained(detector_path, use_fast=True)
        self.person_model = RTDetrForObjectDetection.from_pretrained(detector_path, device_map=self.device)
        self.person_model.eval()

        # --- 2. Inicializar el Estimador de Pose (ViTPose) ---
        self.__file_exists(estimator_path)
        self.pose_image_processor = AutoProcessor.from_pretrained(estimator_path, use_fast=True)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(estimator_path, device_map=self.device)
        self.pose_model.eval()
    
    def __file_exists(self, file_path):
        """Verifica si un archivo o directorio existe."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")

    def __run_two_step_inference(self, frame):
        """
        Ejecuta el flujo de inferencia de dos pasos (RTDetr + ViTPose).
        
        Args:
            frame (np.ndarray): Imagen/frame en formato BGR de OpenCV.
            
        Returns:
            int: Número de personas cuya pose fue estimada (0 si no se detectó ninguna).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        height, width = frame.shape[:2]
        device = self.device

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
            return 0

        # --- 2. Estimación de Pose (ViTPose) ---
        boxes_for_pose = [person_boxes_coco.tolist()] 
        
        inputs_pose = self.pose_image_processor(pil_image, boxes=boxes_for_pose, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=boxes_for_pose)
        
        # El número de sets de keypoints predichos es el número de personas con pose estimada
        return len(pose_results[0])

    def evaluate_image(self, image, threshold, results):
        """
        Evalúa la latencia de inferencia para una imagen usando ViTPose.
        """
        image_path = os.path.join(self.images_path, image)
        self.__file_exists(image_path)

        # Leer imagen
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Medir latencia de ViTPose + RT-DETR
        start_time = time.time()
        # La inferencia completa (detección + pose) es la latencia
        _ = self.__run_two_step_inference(img) 
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Almacenar resultados
        results.append({
            'image': image,
            'image_size': f"{width}x{height}",
            'threshold': threshold,
            'latency_ms': latency_ms
        })

        return results
    
    def evaluate_video(self, videos_path, threshold, results, frame_skip=1, max_frames=None):
        """
        Evalúa la latencia de inferencia para un video grabado (archivo) usando ViTPose.
        """
        video = videos_path # Asumimos que videos_path es la ruta completa
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
        total_frames_to_process = total_frames_file // frame_skip
        if max_frames is not None:
            total_frames_to_process = min(total_frames_to_process, max_frames)
        
        # Variables para métricas
        frame_latencies = []
        frame_counter = 0 # Contador de frames leídos del archivo
        
        # Procesar frames con barra de progreso
        pbar = tqdm(total=total_frames_to_process, desc=f"Procesando {os.path.basename(videos_path)} (th:{threshold})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Saltar frames según frame_skip
            if frame_counter % frame_skip == 0:
                
                # Limitar el número total de frames procesados
                if max_frames is not None and len(frame_latencies) >= max_frames:
                    break
                
                # Medir latencia por frame con ViTPose
                start_time = time.time()
                _ = self.__run_two_step_inference(frame)
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
                pbar.update(1)

            frame_counter += 1
        
        pbar.close()
        cap.release()
        return results