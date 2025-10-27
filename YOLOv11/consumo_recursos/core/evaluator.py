import os
import time
import torch
import cv2 # Necesario para video
from ultralytics import YOLO
from tqdm import tqdm # Necesario para la barra de progreso
from .resource_monitor import ResourceMonitor

class Evaluator:
    """Clase para evaluación de modelos con soporte mejorado para Apple Silicon"""
    
    def __init__(self, basepath, images_path, labels_path, videos_path, model_name='yolo11n-pose.pt'):
        self.basepath = basepath
        self.model_path = os.path.join(basepath, model_name)
        self.model = YOLO(self.model_path)
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        self.monitor = ResourceMonitor()
    
    # --- Métodos de Ayuda (Requieren ser incluidos) ---
    
    def _validate_paths(self, image_path, label_path):
        """Valida la existencia de los archivos"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Etiquetas no encontradas: {label_path}")
    
    def _prepare_device(self, device_type):
        """Configura el dispositivo de ejecución con soporte para Apple Silicon"""
        if device_type == 'gpu':
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            print("GPU no disponible, usando CPU")
        return torch.device("cpu")
    
    def _calculate_deltas(self, before, after):
        """Calcula diferencias en el uso de recursos con soporte para Apple Silicon"""
        # Este método estaba en el código anterior y es llamado por run_video_inference.
        deltas = {
            'cpu_percent': after['cpu']['usage_percent'] - before['cpu']['usage_percent'],
            'ram_mb': after['ram']['rss_mb'] - before['ram']['rss_mb'],
        }
        
        if before['gpu'] and after['gpu']:
            # Manejo especial para diferentes tipos de GPU
            if before['gpu'].get('gpu_type') == 'mps':
                deltas.update({
                    'gpu_memory_used_mb': after['gpu']['memory_used_mb'] - before['gpu']['memory_used_mb'],
                    'gpu_utilization': after['gpu']['utilization_percent'] - before['gpu']['utilization_percent']
                })
            elif before['gpu'].get('gpu_type') == 'cuda':
                # Usar el máximo entre diferentes métricas de memoria
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
        # Este método también es llamado por run_video_inference.
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

    # ... (Si tienes el método run_inference completo, insértalo aquí)

    # --- Método modificado con TQDM (run_video_inference) ---
    def run_video_inference(self, videos_path, threshold, results, device_type='cpu', frame_skip=1, max_frames=None):
        """Ejecuta la inferencia en un video y almacena resultados frame por frame (con TQDM)"""
        
        # Preparar rutas y validar
        video_path = os.path.join(self.videos_path, videos_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        # Configurar dispositivo
        device = self._prepare_device(device_type)
        
        # --- Obtener total de frames para TQDM ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        total_frames_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release() # Liberar el capturador rápidamente
        
        # Calcular total de iteraciones que TQDM debe rastrear
        frames_available_to_process = total_frames_file // frame_skip
        total_to_process = frames_available_to_process
        if max_frames is not None:
            total_to_process = min(frames_available_to_process, max_frames)
        
        # Medición de recursos inicial
        system_info = self.monitor.get_system_info()
        
        # Variables para contadores
        frame_number_read = 0 
        processed_frames_count = 0 
        
        # Inicializar la barra de progreso
        pbar = tqdm(total=total_to_process, desc=f"Video: {videos_path} - {device_type.upper()}")
        
        # Ejecutar inferencia en el video (retorna un GENERADOR)
        for result in self.model.track(
            source=video_path,
            device=device,
            conf=threshold,
            stream=True,
            persist=True,
            verbose=False
        ):
            frame_number_read += 1
            
            # --- Lógica de Salto (Frame Skip) ---
            if frame_number_read % frame_skip != 0:
                continue
            
            # --- Lógica de Límite (Max Frames) ---
            if max_frames is not None and processed_frames_count >= max_frames:
                break
                
            # Medición de recursos por frame
            frame_resources_before = self.monitor.measure()
            frame_start_time = time.perf_counter()
            
            # Procesar frame (la inferencia ya ocurrió en el generador)
            
            # Medición post-procesamiento del frame
            frame_latency = time.perf_counter() - frame_start_time
            frame_resources_after = self.monitor.measure()
            
            # Almacenar resultados del frame
            frame_result = {
                'frame_number': frame_number_read,
                'cpu': system_info['cpu'],
                'ram': system_info['ram'],
                'gpu': system_info['gpu'],
                'device': str(device),
                'video': os.path.basename(videos_path),
                'threshold': threshold,
                'image_size': result.orig_shape if hasattr(result, 'orig_shape') else None,
                'latency': frame_latency,
                'fps': 1 / frame_latency if frame_latency > 0 else float('inf'),
                'resources_before': frame_resources_before,
                'resources_after': frame_resources_after,
                'delta': self._calculate_deltas(frame_resources_before, frame_resources_after),
                'gpu_metrics': self._extract_gpu_metrics(frame_resources_before, frame_resources_after)
            }
            results.append(frame_result)
            
            # --- Actualizar contadores y TQDM ---
            processed_frames_count += 1
            pbar.update(1)
            
        pbar.close()
        return results