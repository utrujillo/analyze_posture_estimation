import os
import time
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
# Asumimos que esta clase es externa y funcional
from .resource_monitor import ResourceMonitor 

class Evaluator:
    """Clase para evaluación de modelos ViTPose (Transformers) con métricas de recursos."""
    
    # NOTA: Los argumentos config_file y checkpoint_file han sido reemplazados por detector_path y estimator_path.
    def __init__(self, basepath, images_path, labels_path, videos_path, detector_path, estimator_path):
        """
        Inicializa el evaluador con ViTPose.
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        self.monitor = ResourceMonitor()
        self.device = torch.device('cpu') 
        
        print("Cargando modelos ViTPose y RTDetr...")

        # --- 1. Inicializar el Detector de Personas (RT-DETR) ---
        self._validate_paths(detector_path)
        self.person_image_processor = AutoProcessor.from_pretrained(detector_path, use_fast=True)
        self.person_model = RTDetrForObjectDetection.from_pretrained(detector_path, device_map='cpu')
        self.person_model.eval()

        # --- 2. Inicializar el Estimador de Pose (ViTPose) ---
        self._validate_paths(estimator_path)
        self.pose_image_processor = AutoProcessor.from_pretrained(estimator_path, use_fast=True)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(estimator_path, device_map='cpu')
        self.pose_model.eval()
    
    def _validate_paths(self, file_path, label_path=None):
        """Valida la existencia de los archivos/directorios."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo/Directorio no encontrado: {file_path}")
        if label_path and not os.path.exists(label_path):
            raise FileNotFoundError(f"Etiquetas no encontradas: {label_path}")
    
    def _prepare_device(self, device_type):
        """Configura el dispositivo de ejecución y mueve ambos modelos si es necesario."""
        
        if device_type == 'gpu':
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                print("GPU no disponible (CUDA/MPS), usando CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        # Mover RTDetr
        current_det_device = next(self.person_model.parameters()).device
        if current_det_device != device:
            self.person_model.to(device)

        # Mover ViTPose
        current_pose_device = next(self.pose_model.parameters()).device
        if current_pose_device != device:
            self.pose_model.to(device)
            
        self.device = device 
        return device
    
    def __run_two_step_inference(self, frame, device):
        """Ejecuta el flujo de inferencia de dos pasos (RTDetr + ViTPose)."""
        
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
            return 0, None

        # --- 2. Estimación de Pose (ViTPose) ---
        boxes_for_pose = [person_boxes_coco.tolist()] 
        
        inputs_pose = self.pose_image_processor(pil_image, boxes=boxes_for_pose, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=boxes_for_pose)
        
        num_keypoints_sets = len(pose_results[0]) if pose_results[0] else 0
        keypoints_data = pose_results[0][0]['keypoints'].cpu().numpy() if num_keypoints_sets > 0 else None

        return num_keypoints_sets, keypoints_data

    # ---------------------------------------------------------------------
    # MÉTODOS AUXILIARES DE RECURSOS (Mantenidos)
    # ---------------------------------------------------------------------

    def _calculate_deltas(self, before, after):
        """Calcula diferencias en el uso de recursos con soporte para Apple Silicon"""
        deltas = {
            'cpu_percent': after['cpu']['usage_percent'] - before['cpu']['usage_percent'],
            'ram_mb': after['ram']['rss_mb'] - before['ram']['rss_mb'],
        }
        
        if before['gpu'] and after['gpu']:
            if before['gpu'].get('gpu_type') == 'mps':
                deltas.update({
                    'gpu_memory_used_mb': after['gpu']['memory_used_mb'] - before['gpu']['memory_used_mb'],
                    'gpu_utilization': after['gpu']['utilization_percent'] - before['gpu']['utilization_percent']
                })
            elif before['gpu'].get('gpu_type') == 'cuda':
                gpu_mem_deltas = [
                    after['gpu'].get('memory_used_mb', 0) - before['gpu'].get('memory_used_mb', 0),
                    after['gpu'].get('memory_used_mb_nvml', 0) - before['gpu'].get('memory_used_mb_nvml', 0)
                ]
                deltas.update({
                    'gpu_memory_used_mb': max(gpu_mem_deltas),
                    'gpu_utilization': after['gpu'].get('utilization_percent', 0) - before['gpu'].get('utilization_percent', 0)
                })
        
        return deltas
    
    def _extract_gpu_metrics(self, before, after):
        """Extrae métricas específicas de GPU para análisis detallado"""
        if not before['gpu'] or not after['gpu']:
            return None
            
        metrics = {
            'before': before['gpu'],
            'after': after['gpu'],
            'type': before['gpu'].get('gpu_type', 'unknown')
        }
        
        if metrics['type'] == 'mps':
            metrics.update({
                'memory_change_mb': after['gpu']['memory_used_mb'] - before['gpu']['memory_used_mb'],
                'utilization_change': after['gpu']['utilization_percent'] - before['gpu']['utilization_percent']
            })
        
        return metrics

    # ---------------------------------------------------------------------
    # MÉTODOS DE INFERENCIA DE IMAGEN
    # ---------------------------------------------------------------------

    def run_inference(self, image, threshold, results, device_type='cpu'):
        """
        Ejecuta la inferencia de ViTPose en una imagen con métricas de recursos.
        Este método reemplaza la funcionalidad de on_cpu y on_gpu.
        """
        image_path = os.path.join(self.images_path, image)
        base_name, ext = os.path.splitext(image)
        label_path = os.path.join(self.labels_path, base_name + '.txt')
        
        try:
            self._validate_paths(image_path, label_path)
            device = self._prepare_device(device_type)
            
            img = cv2.imread(image_path)
            if img is None:
                raise IOError("No se pudo leer la imagen con OpenCV.")

            system_info = self.monitor.get_system_info()
            resources_before = self.monitor.measure()
            start_time = time.perf_counter()
            
            # Ejecutar inferencia de dos pasos
            keypoints_detected, _ = self.__run_two_step_inference(img, device)
            
            elapsed_time = time.perf_counter() - start_time
            resources_after = self.monitor.measure()
            
            image_size = f"{img.shape[1]}x{img.shape[0]}"
            result_entry = {
                'system': system_info['system'],
                'cpu': system_info['cpu'],
                'ram': system_info['ram'],
                'gpu': system_info['gpu'],
                'device': str(device),
                'image': image,
                'threshold': threshold,
                'image_size': image_size,
                'latency': elapsed_time,
                'fps': 1 / elapsed_time if elapsed_time > 0 else float('inf'),
                'keypoints_detected': keypoints_detected,
                'resources_before': resources_before,
                'resources_after': resources_after,
                'delta': self._calculate_deltas(resources_before, resources_after),
                'gpu_metrics': self._extract_gpu_metrics(resources_before, resources_after)
            }
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error durante inferencia de imagen {image}: {str(e)}")
            results.append({'image': image, 'error': str(e), 'success': False})
        
        return results

    # ---------------------------------------------------------------------
    # MÉTODO DE INFERENCIA DE VIDEO
    # ---------------------------------------------------------------------

    def run_video_inference(self, videos_path, threshold, results, device_type='cpu', frame_skip=1, max_frames=None):
        """
        Ejecuta la inferencia de ViTPose en un video frame por frame con barra de progreso.
        """
        
        video_path = videos_path 
        self._validate_paths(video_path)
        
        try:
            device = self._prepare_device(device_type)
        except RuntimeError as e:
            print(f"Advertencia: {e}")
            return results
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # 1. Propiedades y cálculo de frames
        video_name = os.path.basename(videos_path)
        total_frames_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_size = f"{width}x{height}"
        
        total_frames_to_process = total_frames_file // frame_skip
        if max_frames is not None:
            total_frames_to_process = min(total_frames_to_process, max_frames)
            
        processed_frames_count = 0
        frame_counter = 0
        system_info = self.monitor.get_system_info()
        
        # 2. Bucle de procesamiento de frames con tqdm
        pbar = tqdm(total=total_frames_to_process, desc=f"{str(device).upper()}: Procesando ({video_name}) (th:{threshold})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_counter % frame_skip == 0:
                
                if max_frames is not None and processed_frames_count >= max_frames:
                    break
                
                # Medición y Inferencia
                frame_resources_before = self.monitor.measure()
                start_time = time.perf_counter()
                
                keypoints_detected, _ = self.__run_two_step_inference(frame, device)
                
                frame_latency = time.perf_counter() - start_time
                frame_resources_after = self.monitor.measure()
                
                fps_inference = 1 / frame_latency if frame_latency > 0 else float('inf')
                processed_frames_count += 1
                
                # 3. Almacenar resultados
                results.append({
                    'frame_number': frame_counter,
                    'system': system_info['system'],
                    'device': str(device),
                    'video': video_name,
                    'video_fps': fps_video, 
                    'frame_skip': frame_skip, 
                    'image_size': image_size,
                    'threshold': threshold,
                    'latency': frame_latency,
                    'fps': fps_inference,
                    'keypoints_detected': keypoints_detected,
                    'resources_before': frame_resources_before,
                    'resources_after': frame_resources_after,
                    'delta': self._calculate_deltas(frame_resources_before, frame_resources_after),
                    'gpu_metrics': self._extract_gpu_metrics(frame_resources_before, frame_resources_after),
                })
                pbar.update(1)

            frame_counter += 1
        
        cap.release()
        pbar.close()
        return results