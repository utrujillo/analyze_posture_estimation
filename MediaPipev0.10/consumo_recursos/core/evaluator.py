import os
import time
import cv2
import torch
import mediapipe as mp
from tqdm import tqdm
from .resource_monitor import ResourceMonitor

class Evaluator:
    """Clase para evaluación de modelos de pose estimation usando MediaPipe"""
    
    def __init__(self, basepath, images_path, labels_path, videos_path):
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        self.monitor = ResourceMonitor()
        
        # Inicializar MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0: Lite, 1: Full, 2: Heavy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mapeo de keypoints
        self.keypoint_mapping = {
            0: self.mp_pose.PoseLandmark.NOSE,           # Nose
            1: self.mp_pose.PoseLandmark.LEFT_EYE,       # Left-eye
            2: self.mp_pose.PoseLandmark.RIGHT_EYE,      # Right-eye
            3: self.mp_pose.PoseLandmark.LEFT_EAR,      # Left-ear
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
    
    def _process_mediapipe_results(self, results, image_shape):
        """Procesa los resultados de MediaPipe para extraer keypoints"""
        if not results.pose_landmarks:
            return None
            
        pose_landmarks = results.pose_landmarks.landmark
        keypoints = []
        
        for i in range(len(self.keypoint_mapping)):
            landmark = pose_landmarks[self.keypoint_mapping[i]]
            keypoints.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z if hasattr(landmark, 'z') else 0,
                'visibility': landmark.visibility
            })
        
        return {
            'keypoints': keypoints,
            'image_width': image_shape[1],
            'image_height': image_shape[0]
        }
    
    def run_inference(self, image, threshold, results, device_type='cpu'):
        """Ejecuta la inferencia con MediaPipe"""
        try:
            # Preparar rutas y validar
            image_path = os.path.join(self.images_path, image)
            label_path = os.path.join(self.labels_path, image.replace('.jpg', '.txt'))
            self._validate_paths(image_path, label_path)
            
            # Configurar dispositivo (MediaPipe usa su propia configuración de GPU)
            device = self._prepare_device(device_type)
            
            # Medición de recursos
            system_info = self.monitor.get_system_info()
            resources_before = self.monitor.measure()
            start_time = time.perf_counter()
            
            # Leer imagen y convertir a RGB
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ejecutar inferencia con MediaPipe
            mediapipe_results = self.pose.process(img_rgb)
            
            # Medición post-ejecución
            elapsed_time = time.perf_counter() - start_time
            resources_after = self.monitor.measure()
            
            # Procesar resultados
            pose_data = self._process_mediapipe_results(mediapipe_results, img.shape)
            
            # Construir resultados
            result_entry = {
                'system': system_info['system'],
                'cpu': system_info['cpu'],
                'ram': system_info['ram'],
                'gpu': system_info['gpu'],
                'device': str(device),
                'image': image,
                'threshold': threshold,
                'image_size': img.shape[:2],
                'latency': elapsed_time,
                'fps': 1 / elapsed_time if elapsed_time > 0 else float('inf'),
                'resources_before': resources_before,
                'resources_after': resources_after,
                'delta': self._calculate_deltas(resources_before, resources_after),
                'gpu_metrics': self._extract_gpu_metrics(resources_before, resources_after),
                'pose_data': pose_data,
                'keypoints_count': len(pose_data['keypoints']) if pose_data else 0
            }
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error durante inferencia: {str(e)}")
            results.append({
                'image': image,
                'error': str(e),
                'success': False
            })
        
        return results
    
    def _calculate_deltas(self, before, after):
        """Calcula diferencias en el uso de recursos con soporte para Apple Silicon"""
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
    
    def run_video_inference(self, videos_path, threshold, results, device_type='cpu', frame_skip=1, max_frames=None):
        """Ejecuta la inferencia en un video usando MediaPipe con barra de progreso (tqdm)."""
        # Preparar rutas y validar
        video_path = os.path.join(self.videos_path, videos_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        # Configurar dispositivo (MediaPipe usa su propia configuración de GPU)
        device = self._prepare_device(device_type)
        
        # Medición de recursos inicial
        system_info = self.monitor.get_system_info()
        global_resources_before = self.monitor.measure()
        global_start_time = time.perf_counter()
        
        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- Preparación para tqdm ---
        # 1. Definir el número total de frames a iterar.
        #    Si max_frames está definido, limitamos el total para la barra.
        if max_frames is not None:
            # Nota: Aquí se está contando cuántos frames se LEERÁN, no cuántos se PROCESARÁN.
            # total_frames = min(total_frames, max_frames * frame_skip) 
            # (Se mantiene el total de frames original para la barra,
            # y se usa 'max_frames' para el break interno, lo cual es más claro.)
            pass 
        
        processed_frames_count = 0
        video_basename = os.path.basename(videos_path)
        
        # 2. Reemplazamos el 'while cap.isOpened()' por un 'for' con tqdm.
        #    Iteramos sobre el rango total de frames.
        for frame_number in tqdm(range(1, total_frames + 1), 
                                desc=f"{device_type.upper()}: {video_basename}", 
                                unit="frame",
                                total=total_frames):
            
            ret, frame = cap.read()
            
            # Si no se pudo leer el frame (fin de video), salimos del bucle.
            if not ret:
                break
                
            # 3. Mantenemos la lógica de frame_skip
            if frame_number % frame_skip != 0:
                continue
                
            # Medición de recursos por frame
            frame_resources_before = self.monitor.measure()
            frame_start_time = time.perf_counter()
            
            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar frame con MediaPipe
            mediapipe_results = self.pose.process(frame_rgb)
            processed_frames_count += 1
            
            # Medición post-procesamiento del frame
            frame_latency = time.perf_counter() - frame_start_time
            frame_resources_after = self.monitor.measure()
            
            # Procesar resultados
            pose_data = self._process_mediapipe_results(mediapipe_results, frame.shape)
            
            # Almacenar resultados del frame
            frame_result = {
                'frame_number': frame_number,
                'cpu': system_info['cpu'],
                'ram': system_info['ram'],
                'gpu': system_info['gpu'],
                'device': str(device),
                'video': video_basename,
                'threshold': threshold,
                'image_size': frame.shape[:2],
                'latency': frame_latency,
                'fps': 1 / frame_latency if frame_latency > 0 else float('inf'),
                'resources_before': frame_resources_before,
                'resources_after': frame_resources_after,
                'delta': self._calculate_deltas(frame_resources_before, frame_resources_after),
                'pose_data': pose_data,
                'keypoints_count': len(pose_data['keypoints']) if pose_data else 0
            }
            results.append(frame_result)
            
            # Detener si alcanzamos el máximo de frames procesados
            if max_frames is not None and processed_frames_count >= max_frames:
                # 4. Debemos romper el bucle FOR si se cumple la condición
                break 
        
        # Liberar recursos
        cap.release()
        
        return results