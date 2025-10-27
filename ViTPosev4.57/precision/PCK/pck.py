import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class PCK:
    def __init__(self, basepath, images_path, labels_path, detector_path, estimator_path):
        """
        Inicializa el evaluador PCK con ViTPose (Transformers).
        
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
        
    def __calculate_pck(self, true_keypoints, predicted_keypoints, threshold):
        """
        Calcula el PCK dados los puntos clave verdaderos y predichos.
        (Asume que los keypoints de entrada están normalizados a [0, 1])
        """
        # ... (Implementación de __calculate_pck sin cambios) ...
        visible_indices = np.where(true_keypoints[:, 2] == 2)[0]
        true_keypoints_visible = true_keypoints[visible_indices, :2]
        
        # Asegurar que solo comparamos las coordenadas (x, y)
        predicted_keypoints_visible = predicted_keypoints[visible_indices, :2]
        
        distances = np.linalg.norm(true_keypoints_visible - predicted_keypoints_visible, axis=1)
        correct = np.sum(distances < threshold)
        total = len(true_keypoints_visible)
        
        # total_pred se ajusta para reflejar el número de keypoints que estamos comparando
        num_predicted_kpts_to_compare = len(predicted_keypoints) 
        
        pck = (correct / total) * 100 if total > 0 else 0.0
        return pck, len(true_keypoints), total, num_predicted_kpts_to_compare, correct

    def __image_exists(self, image_path, label_path):
        """Verifica si la imagen y el archivo de etiquetas existen."""
        if not os.path.exists(image_path):
            print(f"La imagen {image_path} no existe.")
            return False
        if not os.path.exists(label_path):
            print(f"El archivo de etiquetas {label_path} no existe.")
            return False
        return True

    def __get_true_keypoints(self, label_path, image_width, image_height):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas (en píxeles).
        """
        with open(label_path, 'r') as file:
            line = file.readline().strip()
            parts = line.split()

        true_keypoints = []
        for i in range(5, len(parts), 3):
            # Se convierten las coordenadas normalizadas a píxeles
            x = float(parts[i]) * image_width
            y = float(parts[i+1]) * image_height
            vis = int(parts[i+2])
            true_keypoints.append([x, y, vis])
        return np.array(true_keypoints)

    def __get_inference(self, image_path, true_keypoints, image_name, results, threshold):
        """
        Realiza la inferencia con ViTPose y calcula el PCK.
        
        :param true_keypoints: Coordenadas verdaderas de los keypoints (en píxeles)
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error al cargar la imagen con PIL: {e}")
            return results
        
        width, height = pil_image.size
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
            print(f"No se detectaron personas en la imagen {image_name}.")
            return results
            
        # --- 2. Estimación de Pose (ViTPose) ---
        # Tomaremos la primera persona detectada para la estimación:
        boxes_for_pose = [person_boxes_coco[0:1].tolist()] 
        
        inputs_pose = self.pose_image_processor(pil_image, boxes=boxes_for_pose, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=boxes_for_pose)
        
        if not pose_results[0]:
             print(f"ViTPose no predijo keypoints para la persona detectada en {image_name}.")
             return results

        # Extraer keypoints predichos (en píxeles) para la primera persona
        pred_keypoints_pixels = pose_results[0][0]['keypoints'].cpu().numpy() # N x 2
        pred_scores = pose_results[0][0]['scores'].cpu().numpy()

        # --- 3. Normalización y Cálculo PCK ---
        
        # 1. Normalizar keypoints verdaderos (que están en píxeles) a [0,1]
        true_keypoints_normalized = true_keypoints.copy()
        true_keypoints_normalized[:, 0] /= width
        true_keypoints_normalized[:, 1] /= height
        
        # 2. Normalizar keypoints predichos (en píxeles) a [0,1]
        
        # Es crucial alinear los arrays para la comparación PCK (asumiendo COCO 17kpts)
        num_true_kpts = true_keypoints_normalized.shape[0]
        
        # Inicializar array predicho normalizado y alineado con los verdaderos
        pred_keypoints_aligned_normalized = np.zeros((num_true_kpts, 2))
        
        # Truncar o alinear si el número de predichos no coincide con el de la etiqueta (N_true)
        num_pred_kpts = pred_keypoints_pixels.shape[0]

        if num_pred_kpts > 0:
            kpts_to_align = pred_keypoints_pixels[:num_true_kpts, :2]
            
            # Normalizar los kpts que se van a alinear
            pred_keypoints_aligned_normalized[:kpts_to_align.shape[0], 0] = kpts_to_align[:, 0] / width
            pred_keypoints_aligned_normalized[:kpts_to_align.shape[0], 1] = kpts_to_align[:, 1] / height
        
        # Calcular PCK
        pck, total_true, visible_true, total_pred_aligned, correct_count = self.__calculate_pck(
            true_keypoints_normalized, pred_keypoints_aligned_normalized, threshold)
        
        print(f'PCK: {pck:.2f}%, Total true: {total_true}, Visible true: {visible_true}, '
              f'Total pred (modelo): {num_pred_kpts}, Correctos: {correct_count}')

        results.append({
            'nombre_imagen': image_name,
            'threshold': threshold,
            'image_size': f"{width}x{height}",
            'cantidad_true_keypoints': total_true,
            'true_keypoints_visible': visible_true,
            'cantidad_pred_keypoints': num_pred_kpts, # Total del modelo (sin alinear)
            'pck': pck
        })
        
        return results

    def evaluate_image(self, image, threshold, results):
        """
        Evalúa una imagen y calcula el PCK.
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        if not self.__image_exists(image_path, label_path):
            return results
        
        # Obtener dimensiones de la imagen
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Obtener keypoints verdaderos en píxeles
        true_keypoints = self.__get_true_keypoints(label_path, width, height)
        return self.__get_inference(image_path, true_keypoints, image, results, threshold)

    # Las funciones draw_original_keypoints y draw_predicted_keypoints se omiten o se migran si es necesario
    # Aquí solo migraremos draw_predicted_keypoints para la visualización

    def draw_original_keypoints(self, image_path, label_path):
        """
        Dibuja los keypoints originales en la imagen. (Sin cambios)
        """
        # ... (implementación de draw_original_keypoints sin cambios) ...
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
        
        height, width, _ = image.shape
        img_with_original_keypoints = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label in labels:
            parts = list(map(float, label.split()))
            class_id, x_center, y_center, w, h = parts[:5]
            keypoints = parts[5:]

            # Convertir coordenadas normalizadas a píxeles
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Dibujar el rectángulo del bounding box
            cv2.rectangle(img_with_original_keypoints, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Procesar los keypoints
            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                x = int(x * width)
                y = int(y * height)
                size = 5

                if visibility == 2:  # Keypoint visible
                    cv2.circle(img_with_original_keypoints, (x, y), size, (254, 250, 224), -1)
                elif visibility == 1:  # Keypoint no visible
                    cv2.line(img_with_original_keypoints, (x - size, y - size), (x + size, y + size), (230, 57, 70), 2)
                    cv2.line(img_with_original_keypoints, (x + size, y - size), (x - size, y + size), (230, 57, 70), 2)
                else:
                    # Dibujar un triángulo rojo
                    triangle_points = np.array([
                        [x, y - size],  # Punto superior
                        [x - size, y + size],  # Punto inferior izquierdo
                        [x + size, y + size],  # Punto inferior derecho
                    ], dtype=np.int32)
                    cv2.drawContours(img_with_original_keypoints, [triangle_points], 0, (144, 169, 85), -1)

        return img_with_original_keypoints

    def draw_predicted_keypoints(self, image_path):
        """
        Dibuja los keypoints predichos en la imagen usando ViTPose.
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

        image_rgb_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        width, height = image_rgb_pil.size
        device = self.device
        
        # --- 1. Detección de Personas (RT-DETR) ---
        inputs_det = self.person_image_processor(images=image_rgb_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_det = self.person_model(**inputs_det)

        results_det = self.person_image_processor.post_process_object_detection(
            outputs_det, target_sizes=torch.tensor([(height, width)]), threshold=0.5
        )
        person_boxes = results_det[0]["boxes"][results_det[0]["labels"] == 0].cpu().numpy()

        person_boxes_coco = person_boxes.copy()
        person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]
        person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]
        
        if len(person_boxes_coco) == 0:
            print("No se detectaron personas para la estimación de pose.")
            return image_bgr, None

        # --- 2. Estimación de Pose (ViTPose) ---
        boxes_for_pose = [person_boxes_coco[0:1].tolist()] 
        
        inputs_pose = self.pose_image_processor(image_rgb_pil, boxes=boxes_for_pose, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=boxes_for_pose)
        
        if pose_results[0]:
            keypoints = pose_results[0][0]['keypoints'].cpu().numpy()
            scores = pose_results[0][0]['scores'].cpu().numpy()
            
            # Dibujar keypoints con alta confianza
            for kpt, score in zip(keypoints, scores):
                if score > 0.3:  # Umbral de confianza
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(image_bgr, (x, y), 5, (0, 255, 0), -1)

        return image_bgr, pose_results