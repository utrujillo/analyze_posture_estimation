import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class mAP:
    def __init__(self, basepath, images_path, labels_path, detector_path, estimator_path):
        """
        Inicializa la clase mAP con ViTPose (Transformers).
        
        :param basepath: Ruta base donde se encuentran los archivos.
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

    def get_true_keypoints(self, label_path):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas (NORMALIZADAS).
        """
        with open(label_path, 'r') as file:
            line = file.readline().strip()
            parts = line.split()

        true_keypoints = []
        for i in range(5, len(parts), 3):
            x = float(parts[i])
            y = float(parts[i + 1])
            vis = int(parts[i + 2])
            true_keypoints.append([x, y, vis])
        return np.array(true_keypoints)

    def draw_original_keypoints(self, image_path, label_path):
        """
        Dibuja los keypoints originales en la imagen. (Sin cambios)
        """
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        img_with_original_keypoints = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label in labels:
            parts = list(map(float, label.split()))
            class_id, x_center, y_center, w, h = parts[:5]
            keypoints = parts[5:]

            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            cv2.rectangle(img_with_original_keypoints, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                if visibility > 0:
                    x = int(x * width)
                    y = int(y * height)
                    cv2.circle(img_with_original_keypoints, (x, y), 5, (255, 0, 0), -1)

        return img_with_original_keypoints

    def draw_predicted_keypoints(self, image_path):
        """
        Dibuja los keypoints predichos en la imagen usando ViTPose.
        
        :param image_path: Ruta de la imagen.
        :return: Imagen con los keypoints predichos (np array BGR) y resultados (lista de dicts de Hugging Face).
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

        # Convertir cajas de VOC a COCO
        person_boxes_coco = person_boxes.copy()
        person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]
        person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]
        
        if len(person_boxes_coco) == 0:
            return image_bgr, [] # Retornamos lista vacía en lugar de un objeto de MMPose

        # --- 2. Estimación de Pose (ViTPose) ---
        boxes_for_pose = [person_boxes_coco[0:1].tolist()] # Solo tomamos la primera caja
        
        inputs_pose = self.pose_image_processor(image_rgb_pil, boxes=boxes_for_pose, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_pose = self.pose_model(**inputs_pose)

        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs_pose, boxes=boxes_for_pose)
        
        inferred_instances = []
        
        if pose_results[0]:
            # Asumimos una única persona para la métrica
            person_result = pose_results[0][0] 
            keypoints = person_result['keypoints'].cpu().numpy()
            scores = person_result['scores'].cpu().numpy()
            
            # Dibujar y recopilar la primera instancia
            x1, y1, w, h = [int(val) for val in person_boxes_coco[0]]
            cv2.rectangle(image_bgr, (x1, y1), (x1 + w, y1 + h), (255, 100, 0), 2)

            for kpt, score in zip(keypoints, scores):
                if score > 0.3:
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(image_bgr, (x, y), 5, (0, 255, 0), -1)

            inferred_instances.append({
                'keypoints': keypoints, # Coordenadas en píxeles
                'scores': scores,
                'bbox': person_boxes_coco[0] # [x, y, w, h] COCO format (píxeles)
            })
            
        return image_bgr, inferred_instances

    def normalize_keypoints(self, keypoints, image_width, image_height):
        """
        Normaliza las coordenadas de los keypoints.
        """
        keypoints[:, 0] /= image_width
        keypoints[:, 1] /= image_height
        return keypoints

    def calculate_oks(self, true_keypoints, pred_keypoints, scale):
        """
        Calcula el OKS (Object Keypoint Similarity). (Sin cambios)
        """
        k = len(true_keypoints)
        oks = np.zeros(k)

        # Aseguramos que el array de predichos tenga el mismo tamaño y orden
        num_true_kpts = true_keypoints.shape[0]
        if pred_keypoints.shape[0] < num_true_kpts:
             temp_pred_kpts = np.zeros((num_true_kpts, 2))
             temp_pred_kpts[:pred_keypoints.shape[0]] = pred_keypoints
             pred_keypoints = temp_pred_kpts
        elif pred_keypoints.shape[0] > num_true_kpts:
             pred_keypoints = pred_keypoints[:num_true_kpts]
        
        
        for i in range(k):
            x_true, y_true, vis_true = true_keypoints[i]
            x_pred, y_pred = pred_keypoints[i] # Ahora pred_keypoints tiene N_true filas

            # Solo comparar si el keypoint verdadero es visible
            if vis_true > 0:
                d = (x_true - x_pred) ** 2 + (y_true - y_pred) ** 2
                oks[i] = np.exp(-d / (2 * scale ** 2))

        return oks

    def calculate_ap(self, true_keypoints, pred_keypoints, scale, threshold, image, results):
        """
        Calcula el AP (Average Precision). (Sin cambios)
        """
        # ... (implementación de calculate_ap sin cambios) ...
        
        # NOTE: La lógica de alineación se ha movido a calculate_oks.
        oks = self.calculate_oks(true_keypoints, pred_keypoints, scale)

        visible_mask = true_keypoints[:, 2] > 0
        oks = oks[visible_mask]

        correct = oks >= threshold
        num_visible = np.sum(visible_mask)
        true_positives = np.sum(correct)
        false_positives = len(correct) - true_positives # Asume que todos los predichos son comparados (lógica simplificada)
        false_negatives = num_visible - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        ap = precision
        
        print(f'Imagen: {image}, threshold: {threshold}, true_keypoints: {len(true_keypoints)}, \
          num_visible: {num_visible}, true_positives: {true_positives}, false_positives: {false_positives}, \
          false_negatives: {false_negatives}, recall: {recall}, ap: {ap}')
        
        results.append({
            'image': image,
            'threshold': threshold,
            'true_keypoints': len(true_keypoints),
            'num_visible': num_visible,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'recall': recall,
            'ap': ap
        })

        return results

    def evaluate_image(self, image_name, threshold, results):
        """
        Evalúa una imagen y calcula el AP utilizando ViTPose.
        """
        image_path = os.path.join(self.images_path, image_name)
        label_path = os.path.join(self.labels_path, os.path.splitext(image_name)[0] + '.txt')

        # Obtener keypoints verdaderos (NORMALIZADOS)
        true_keypoints = self.get_true_keypoints(label_path)

        # Obtener keypoints predichos con ViTPose
        img_with_predicted_keypoints, inferred_instances = self.draw_predicted_keypoints(image_path)
        
        if not inferred_instances:
            print(f"No se detectó pose en la imagen {image_name}. AP = 0.")
            # Para evitar un error en el cálculo, pasamos un array vacío de predichos
            zero_kpts = np.zeros((len(true_keypoints), 2)) 
            zero_scale = 0.01 # Un valor de escala pequeño para que OKS pueda funcionar
            return self.calculate_ap(true_keypoints, zero_kpts, zero_scale, threshold, image_name, results)

        # Tomamos solo la primera persona detectada para la métrica (asume una sola persona etiquetada)
        instance_data = inferred_instances[0]
        
        pred_keypoints_pixels = instance_data['keypoints'] # En píxeles
        bbox_coco = instance_data['bbox'] # [x, y, w, h] en píxeles

        image_height, image_width, _ = img_with_predicted_keypoints.shape
        
        # 1. Normalizar keypoints predichos
        pred_keypoints_normalized = self.normalize_keypoints(pred_keypoints_pixels, image_width, image_height)

        # 2. Calcular la escala (normalizada)
        area_pixels = bbox_coco[2] * bbox_coco[3]
        area_normalized = area_pixels / (image_width * image_height)
        
        # Usamos la raíz cuadrada del área normalizada, ajustada por el factor original
        scale = np.sqrt(area_normalized) / 10 

        # 3. Calcular el AP
        return self.calculate_ap(true_keypoints, pred_keypoints_normalized, scale, threshold, image_name, results)