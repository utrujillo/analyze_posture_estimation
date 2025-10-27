import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class DropRate:
    def __init__(self, basepath, images_path, labels_path, detector_path, estimator_path):
        """
        Inicializa el evaluador DropRate con ViTPose (Transformers).
        
        :param basepath: Ruta base del proyecto
        :param images_path: Ruta de las imágenes
        :param labels_path: Ruta de las etiquetas
        :param detector_path: Ruta local del modelo RTDetr para detección de personas.
        :param estimator_path: Ruta local del modelo ViTPose para estimación de pose.
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Cargando modelos en el dispositivo: {self.device}")

        # --- 1. Inicializar el Detector de Personas (RT-DETR) ---
        self.person_image_processor = AutoProcessor.from_pretrained(detector_path, use_fast=True)
        self.person_model = RTDetrForObjectDetection.from_pretrained(detector_path, device_map=self.device)
        self.person_model.eval()

        # --- 2. Inicializar el Estimador de Pose (ViTPose) ---
        self.pose_image_processor = AutoProcessor.from_pretrained(estimator_path, use_fast=True)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(estimator_path, device_map=self.device)
        self.pose_model.eval()
    
    def __image_exists(self, image_path, label_path):
        """Verifica si la imagen y el archivo de etiquetas existen."""
        if not os.path.exists(image_path):
            print(f"La imagen {image_path} no existe.")
            return False
        if not os.path.exists(label_path):
            print(f"El archivo de etiquetas {label_path} no existe.")
            return False
        return True
    
    def get_true_keypoints(self, label_path):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas.
        NOTA: Retorna keypoints en formato NORMALIZADO [0, 1] y visibilidad.
        """
        with open(label_path, 'r') as file:
            line = file.readline().strip()
            parts = line.split()

        true_keypoints = []
        for i in range(5, len(parts), 3):
            x = float(parts[i])      # Coordenada x normalizada
            y = float(parts[i + 1])  # Coordenada y normalizada
            vis = int(parts[i + 2])
            true_keypoints.append([x, y, vis])
        return np.array(true_keypoints)
    
    def get_pred_keypoints(self, image_path):
        """
        Obtiene las coordenadas predichas de los keypoints para una imagen usando ViTPose.
        
        :param image_path: Ruta de la imagen.
        :return: Coordenadas predichas de los keypoints (normalizadas, Nx3: x, y, score), ancho y alto de la imagen.
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error al cargar la imagen con PIL: {e}")
            return None, None, None
        
        img_width, img_height = pil_image.size
        device = self.device
        
        # --- 1. Detección de Personas (RT-DETR) ---
        inputs_det = self.person_image_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_det = self.person_model(**inputs_det)

        results_det = self.person_image_processor.post_process_object_detection(
            outputs_det, target_sizes=torch.tensor([(img_height, img_width)]), threshold=0.5
        )
        person_boxes = results_det[0]["boxes"][results_det[0]["labels"] == 0].cpu().numpy()

        # Convertir cajas de VOC a COCO
        person_boxes_coco = person_boxes.copy()
        person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]
        person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]
        
        if len(person_boxes_coco) == 0:
            return None, img_width, img_height
            
        # --- 2. Estimación de Pose (ViTPose) ---
        boxes_for_pose = [person_boxes_coco[0:1].tolist()] # Asumimos una sola persona
        
        inputs_pose = self.pose_image_processor(pil_image, boxes=boxes_for_pose, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=boxes_for_pose)
        
        if not pose_results[0]:
             print(f"ViTPose no predijo keypoints para la persona detectada en {image_path}.")
             return None, img_width, img_height

        # Extraer keypoints (en píxeles) y scores
        pred_keypoints_pixels = pose_results[0][0]['keypoints'].cpu().numpy() # N x 2
        pred_scores = pose_results[0][0]['scores'].cpu().numpy()             # N
        
        # Normalizar coordenadas
        pred_keypoints_normalized = pred_keypoints_pixels.copy()
        pred_keypoints_normalized[:, 0] /= img_width
        pred_keypoints_normalized[:, 1] /= img_height
        
        # Combinar coordenadas normalizadas con scores (como visibilidad/confianza)
        pred_keypoints_with_vis = np.hstack([
            pred_keypoints_normalized, 
            pred_scores.reshape(-1, 1)
        ])
        
        return pred_keypoints_with_vis, img_width, img_height
    
    def get_drop_rate(self, true_keypoints, pred_keypoints, threshold):
        """
        Calcula el drop rate comparando los keypoints reales (normalizados) con los predichos (normalizados).
        """
        # 1. Filtrar keypoints visibles (estado == 2)
        visible_mask = true_keypoints[:, 2] == 2
        visible_keypoints_true = true_keypoints[visible_mask, :2] # Solo coords (x, y)
        total_visible = len(visible_keypoints_true)

        if total_visible == 0:
            return 0.0, 0, 0

        # 2. Alinear los keypoints predichos con los verdaderos (por índice)
        num_true_kpts = true_keypoints.shape[0]
        
        if pred_keypoints.shape[0] < num_true_kpts:
             # Rellenamos con un valor de 0 para keypoints no predichos (distancia será > 0)
             temp_pred_kpts = np.zeros((num_true_kpts, 3)) 
             temp_pred_kpts[:pred_keypoints.shape[0]] = pred_keypoints
             pred_keypoints_aligned = temp_pred_kpts
        elif pred_keypoints.shape[0] > num_true_kpts:
             # Truncamos los predichos para que coincidan con la etiqueta
             pred_keypoints_aligned = pred_keypoints[:num_true_kpts]
        else:
            pred_keypoints_aligned = pred_keypoints
             
        # Tomar solo las coordenadas (x, y) de los keypoints predichos correspondientes a los visibles
        pred_keypoints_for_comparison = pred_keypoints_aligned[visible_mask, :2]
        
        # 3. Calcular la distancia y la detección
        
        # Calcular la distancia euclidiana entre los puntos alineados
        distances = np.linalg.norm(visible_keypoints_true - pred_keypoints_for_comparison, axis=1)
        
        # Un keypoint se considera detectado si la distancia es menor que el umbral
        detected_count = np.sum(distances < threshold)

        # 4. Calcular el drop rate
        drop_rate = ((total_visible - detected_count) / total_visible) * 100
        return drop_rate, total_visible, detected_count

    def evaluate_image(self, image, threshold, results):
        """
        Evalúa una imagen y calcula el drop rate.
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        if not self.__image_exists(image_path, label_path):
            return results
            
        true_keypoints = self.get_true_keypoints(label_path)
        
        # pred_keypoints ahora incluye el score en la 3ra columna
        pred_keypoints_with_score, image_width, image_height = self.get_pred_keypoints(image_path)
        
        if pred_keypoints_with_score is not None and image_width is not None:
            # Solo pasamos las coordenadas (x, y) al calculador de drop rate
            drop_rate, total_visible, detected_count = self.get_drop_rate(true_keypoints, pred_keypoints_with_score[:, :2], threshold)
            
            results.append({
                'image_name': image,
                'image_size': f"{image_width}x{image_height}",
                'threshold': threshold,
                'true_keypoints_total': len(true_keypoints),
                'pred_keypoints_total': pred_keypoints_with_score.shape[0],
                'total_visible_true': total_visible,
                'detected_count': detected_count,
                'drop_rate': drop_rate
            })
        elif image_width is not None:
             # Caso de no detección: el drop rate es 100% para los keypoints visibles (si los hay)
             total_visible = np.sum(true_keypoints[:, 2] == 2)
             drop_rate = 100.0 if total_visible > 0 else 0.0
                 
             results.append({
                'image_name': image,
                'image_size': f"{image_width}x{image_height}",
                'threshold': threshold,
                'true_keypoints_total': len(true_keypoints),
                'pred_keypoints_total': 0,
                'total_visible_true': total_visible,
                'detected_count': 0,
                'drop_rate': drop_rate
            })
        
        return results