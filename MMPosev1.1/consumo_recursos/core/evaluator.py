import os
import time
import torch
import cv2 # Necesario para el procesamiento de video con MMPose
import numpy as np
from mmpose.apis import init_model, inference_topdown # Importar funciones de MMPose
from mmpose.structures import merge_data_samples # Utilidad para consolidar resultados
from tqdm import tqdm # Se añade para la barra de progreso en video
from .resource_monitor import ResourceMonitor

class Evaluator:
    """Clase para evaluación de modelos MMPose con soporte mejorado para Apple Silicon"""
    
    def __init__(self, basepath, images_path, labels_path, videos_path, config_file, checkpoint_file):
        """
        Inicializa el evaluador con MMPose.
        
        :param basepath: Ruta base del proyecto
        :param images_path: Ruta de las imágenes
        :param labels_path: Ruta de las etiquetas
        :param videos_path: Ruta de los videos
        :param config_file: Archivo de configuración del modelo MMPose (ruta completa o relativa)
        :param checkpoint_file: Checkpoint del modelo MMPose (ruta completa o relativa)
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        self.monitor = ResourceMonitor()
        
        # 1. Inicialización del modelo MMPose
        # Por defecto, inicializamos en CPU, y lo movemos a GPU/MPS en _prepare_device si es necesario
        self.model = init_model(config_file, checkpoint_file, device='cpu') 
    
    def _validate_paths(self, file_path, label_path=None):
        """Valida la existencia de los archivos (imagen/video y etiquetas opcionales)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        if label_path and not os.path.exists(label_path):
            raise FileNotFoundError(f"Etiquetas no encontradas: {label_path}")
    
    def _prepare_device(self, device_type):
        """Configura el dispositivo de ejecución y mueve el modelo MMPose si es necesario."""
        
        # Determinar el dispositivo
        if device_type == 'gpu':
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                print("GPU no disponible (CUDA/MPS), usando CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        # Mover el modelo al dispositivo determinado
        current_device = next(self.model.parameters()).device
        if current_device != device:
            self.model = self.model.to(device)

        return device
    
    # ---------------------------------------------------------------------
    # Métodos auxiliares (sin cambios, mantienen la lógica de YOLO)
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
    # Métodos de Inferencia Actualizados
    # ---------------------------------------------------------------------

    def run_inference(self, image, threshold, results, device_type='cpu'):
        """Ejecuta la inferencia de MMPose en una imagen."""
        
        image_path = os.path.join(self.images_path, image)
        base_name, ext = os.path.splitext(image)
        label_path = os.path.join(self.labels_path, base_name + '.txt')
        
        try:
            # 1. Preparar rutas y validar
            self._validate_paths(image_path, label_path)
            
            # 2. Configurar dispositivo y mover el modelo
            device = self._prepare_device(device_type)
            
            # 3. Lectura de la imagen (MMPose prefiere np.ndarray)
            img = cv2.imread(image_path)
            if img is None:
                raise IOError("No se pudo leer la imagen con OpenCV.")

            # 4. Medición de recursos
            system_info = self.monitor.get_system_info()
            resources_before = self.monitor.measure()
            start_time = time.perf_counter()
            
            # 5. Ejecutar inferencia con MMPose
            pose_results = inference_topdown(self.model, img)
            data_samples = merge_data_samples(pose_results) # Consolida los resultados

            # 6. Medición post-ejecución
            elapsed_time = time.perf_counter() - start_time
            resources_after = self.monitor.measure()
            
            # 7. Construir resultados
            result_entry = {
                'system': system_info['system'],
                'cpu': system_info['cpu'],
                'ram': system_info['ram'],
                'gpu': system_info['gpu'],
                'device': str(device),
                'image': image,
                'threshold': threshold,
                'image_size': img.shape[:2], # Usar el shape de la imagen leída
                'latency': elapsed_time,
                'fps': 1 / elapsed_time if elapsed_time > 0 else float('inf'),
                'keypoints_found': data_samples.pred_instances.keypoints.shape[0] if hasattr(data_samples, 'pred_instances') and data_samples.pred_instances.keypoints.shape else 0,
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

    def run_video_inference(self, videos_path, threshold, results, device_type='cpu', frame_skip=1, max_frames=None):
        """Ejecuta la inferencia de MMPose en un video frame por frame con barra de progreso."""
        
        video_path = os.path.join(self.videos_path, videos_path)
        
        try:
            # 1. Preparar rutas y validar
            self._validate_paths(video_path)
            
            # 2. Configurar dispositivo y mover el modelo
            device = self._prepare_device(device_type)
            
            # 3. Apertura del video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {video_path}")
            
            # 4. Propiedades del video
            video_name = os.path.basename(videos_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 5. Medición de recursos inicial
            system_info = self.monitor.get_system_info()
            
            processed_frames = 0
            
            # 6. Bucle de procesamiento de frames con tqdm
            for frame_count in tqdm(range(total_frames), desc=f"{str(device).upper()}: Procesando {video_path} (th:{threshold})"):
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Saltar frames según frame_skip
                if frame_count % frame_skip != 0:
                    continue
                
                # **Inferencia por Frame**
                
                frame_resources_before = self.monitor.measure()
                frame_start_time = time.perf_counter()
                
                # Ejecutar inferencia con MMPose
                pose_results = inference_topdown(self.model, frame)
                data_samples = merge_data_samples(pose_results)
                
                frame_latency = time.perf_counter() - frame_start_time
                frame_resources_after = self.monitor.measure()
                
                processed_frames += 1
                
                # Almacenar resultados del frame
                frame_result = {
                    'frame_number': frame_count,
                    'system': system_info['system'],
                    'device': str(device),
                    'video': video_name,
                    'threshold': threshold,
                    'image_size': (height, width),
                    'latency': frame_latency,
                    'fps': 1 / frame_latency if frame_latency > 0 else float('inf'),
                    'keypoints_found': data_samples.pred_instances.keypoints.shape[0] if hasattr(data_samples, 'pred_instances') and data_samples.pred_instances.keypoints.shape else 0,
                    'resources_before': frame_resources_before,
                    'resources_after': frame_resources_after,
                    'delta': self._calculate_deltas(frame_resources_before, frame_resources_after),
                    'gpu_metrics': self._extract_gpu_metrics(frame_resources_before, frame_resources_after)
                }
                results.append(frame_result)
                
                # Detener si alcanzamos el máximo de frames procesados
                if max_frames is not None and processed_frames >= max_frames:
                    break
            
            cap.release()
            
        except Exception as e:
            print(f"Error durante inferencia de video {videos_path}: {str(e)}")
            results.append({'video': videos_path, 'error': str(e), 'success': False})
        
        return results