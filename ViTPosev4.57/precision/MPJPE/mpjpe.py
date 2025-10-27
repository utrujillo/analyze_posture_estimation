import os, math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class MPJPE:
    def __init__(self, basepath, images_path, labels_path, detector_path, estimator_path):
        """
        Inicializa el evaluador MPJPE con ViTPose (Transformers).
        
        :param basepath: Ruta base del proyecto
        :param images_path: Ruta de las imágenes
        :param labels_path: Ruta de las etiquetas
        :param detector_path: Ruta local del modelo RTDetr para detección de personas.
        :param estimator_path: Ruta local del modelo ViTPose para estimación de pose.
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        # Usar la GPU si está disponible
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

    def __get_true_keypoints(self, label_path):
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
            y = float(parts[i+1])    # Coordenada y normalizada
            vis = int(parts[i+2])
            true_keypoints.append([x, y, vis])
        return np.array(true_keypoints)
    
    def __euclidean_distance(self, true_keypoints, pred_keypoints):
        """
        Calcula las distancias euclidianas entre keypoints (asume que ambos están en el mismo espacio normalizado).
        """
        # Distancias por fila (por keypoint)
        distances = np.linalg.norm(true_keypoints - pred_keypoints, axis=1)
        return distances.tolist()
    
    def __calculate_mpjpe(self, true_keypoints, pred_keypoints):
        """
        Calcula el MPJPE.
        :param true_keypoints: Coordenadas verdaderas de los keypoints (normalizadas con visibilidad)
        :param pred_keypoints: Coordenadas predichas de los keypoints (normalizadas)
        """
        visible_indices = np.where(true_keypoints[:, 2] == 2)[0]
        true_keypoints_visible = true_keypoints[visible_indices, :2]
        
        # --- Alineación de Keypoints ---
        num_true_kpts = true_keypoints.shape[0]
        
        if pred_keypoints.shape[0] < num_true_kpts:
             # Rellenar con ceros (MPJPE penalizará fuertemente los no detectados)
             temp_pred_kpts = np.zeros((num_true_kpts, 2))
             temp_pred_kpts[:pred_keypoints.shape[0]] = pred_keypoints
             pred_keypoints = temp_pred_kpts
        elif pred_keypoints.shape[0] > num_true_kpts:
             # Truncar (asumiendo que los primeros coinciden)
             pred_keypoints = pred_keypoints[:num_true_kpts]
             
        predicted_keypoints_visible = pred_keypoints[visible_indices, :2]
        # -------------------------------
        
        distances = self.__euclidean_distance(true_keypoints_visible, predicted_keypoints_visible)
        mpjpe = np.mean(distances) if distances else None

        return mpjpe, len(true_keypoints_visible), len(predicted_keypoints_visible)
    
    def __calculate_diff_in_pixels(self, image_width, image_height, mpjpe):
        """
        Convierte el MPJPE (en unidades normalizadas) a píxeles usando la diagonal.
        """
        if mpjpe is None:
            return None
            
        diagonal = math.sqrt(image_width**2 + image_height**2)
        mpjpe_pixels = mpjpe * diagonal
    
        return mpjpe_pixels

    def __get_inference(self, image_path, true_keypoints, image_name, results):
        """
        Realiza la inferencia de ViTPose y calcula el MPJPE.
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error al cargar la imagen con PIL: {e}")
            return results
        
        image_width, image_height = pil_image.size
        device = self.device
        
        # --- 1. Detección de Personas (RT-DETR) ---
        inputs_det = self.person_image_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_det = self.person_model(**inputs_det)

        results_det = self.person_image_processor.post_process_object_detection(
            outputs_det, target_sizes=torch.tensor([(image_height, image_width)]), threshold=0.5
        )
        person_boxes = results_det[0]["boxes"][results_det[0]["labels"] == 0].cpu().numpy()

        # Convertir cajas de VOC a COCO
        person_boxes_coco = person_boxes.copy()
        person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]
        person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]
        
        if len(person_boxes_coco) == 0:
            print(f"No se detectaron personas en la imagen {image_name}.")
            return results
            
        # --- 2. Estimación de Pose (ViTPose) ---
        # Asumimos una sola persona para la evaluación
        boxes_for_pose = [person_boxes_coco[0:1].tolist()] 
        
        inputs_pose = self.pose_image_processor(pil_image, boxes=boxes_for_pose, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=boxes_for_pose)
        
        if not pose_results[0]:
             print(f"ViTPose no predijo keypoints para la persona detectada en {image_name}.")
             return results

        # Extraer keypoints (en píxeles) para la primera persona
        pred_keypoints_pixels = pose_results[0][0]['keypoints'].cpu().numpy() # N x 2
        
        # Normalizar las coordenadas predichas a [0, 1]
        pred_keypoints_normalized = pred_keypoints_pixels.copy()
        pred_keypoints_normalized[:, 0] /= image_width
        pred_keypoints_normalized[:, 1] /= image_height

        # --- 3. Cálculo MPJPE ---
        mpjpe, visible_true_count, visible_pred_aligned_count = self.__calculate_mpjpe(
            true_keypoints, pred_keypoints_normalized)
            
        pixels = self.__calculate_diff_in_pixels(image_width, image_height, mpjpe)
        
        print(f"MPJPE: {mpjpe:.4f} (normalized), {pixels:.2f} (pixels)")

        results.append({
            'nombre_imagen': image_name,
            'image_size': f"{image_width}x{image_height}",
            'cantidad_true_keypoints': len(true_keypoints),
            'true_keypoints_visible': visible_true_count,
            'cantidad_pred_keypoints': pred_keypoints_pixels.shape[0], # Total predicho por el modelo
            'pred_keypoints_visible_comparados': visible_pred_aligned_count,
            'mpjpe_pixels': pixels,
            'mpjpe': mpjpe
        })

        return results
        
    def evaluate_image(self, image, results):
        """
        Evalúa una imagen y calcula el MPJPE.
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        if not self.__image_exists(image_path, label_path):
            return results
        
        # Obtener keypoints verdaderos (normalizados)
        true_keypoints = self.__get_true_keypoints(label_path) 

        return self.__get_inference(image_path, true_keypoints, image, results)