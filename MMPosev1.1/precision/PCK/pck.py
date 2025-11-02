import os
import cv2
import numpy as np
import pandas as pd
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

class PCK:
    def __init__(self, basepath, images_path, labels_path, config_file, checkpoint_file):
        """
        Inicializa el evaluador PCK con MMPose.
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        
        # Inicializar el modelo MMPose
        self.model = init_model(config_file, checkpoint_file, device='cpu')
        
        # Mapeo de keypoints entre tu formato (17) y MMPose (puede ser diferente)
        # Este mapeo depende del modelo específico de MMPose que estés usando
        self.keypoint_mapping = self._get_keypoint_mapping()
        
    def _get_keypoint_mapping(self):
        """
        Define el mapeo entre tus 17 keypoints y los keypoints del modelo MMPose.
        Necesitas ajustar esto según el modelo específico que uses.
        """
        # Mapeo genérico - AJUSTA ESTO SEGÚN TU MODELO MMPose
        # Esto asume que MMPose también usa 17 keypoints pero en orden diferente
        # Si tu modelo de MMPose usa menos keypoints, necesitarás un mapeo diferente
        return list(range(17))  # Mapeo directo por defecto
    
    def __calculate_pck(self, true_keypoints, predicted_keypoints, threshold):
        """
        Calcula el PCK dados los puntos clave verdaderos y predichos.
        """
        # Solo considerar keypoints que existen en ambos conjuntos
        min_keypoints = min(len(true_keypoints), len(predicted_keypoints))
        
        # Usar solo los keypoints comunes
        true_keypoints_common = true_keypoints[:min_keypoints]
        predicted_keypoints_common = predicted_keypoints[:min_keypoints]
        
        # Filtrar solo keypoints visibles
        visible_indices = np.where(true_keypoints_common[:, 2] == 2)[0]
        
        if len(visible_indices) == 0:
            return 0.0, len(true_keypoints), 0, len(predicted_keypoints), 0
            
        true_keypoints_visible = true_keypoints_common[visible_indices, :2]
        predicted_keypoints_visible = predicted_keypoints_common[visible_indices, :2]
        
        distances = np.linalg.norm(true_keypoints_visible - predicted_keypoints_visible, axis=1)
        correct = np.sum(distances < threshold)
        total = len(true_keypoints_visible)
        
        pck = (correct / total) * 100 if total > 0 else 0.0
        return pck, len(true_keypoints), len(true_keypoints_visible), len(predicted_keypoints), len(predicted_keypoints_visible)

    def __image_exists(self, image_path, label_path):
        """
        Verifica si la imagen y el archivo de etiquetas existen.
        """
        if not os.path.exists(image_path):
            print(f"La imagen {image_path} no existe.")
            return False
        if not os.path.exists(label_path):
            print(f"El archivo de etiquetas {label_path} no existe.")
            return False
        return True

    def __get_true_keypoints(self, label_path, image_width, image_height):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas.
        """
        try:
            with open(label_path, 'r') as file:
                line = file.readline().strip()
                parts = line.split()

            true_keypoints = []
            for i in range(5, len(parts), 3):
                x = float(parts[i]) * image_width
                y = float(parts[i+1]) * image_height
                vis = int(parts[i+2])
                true_keypoints.append([x, y, vis])
            return np.array(true_keypoints)
        except Exception as e:
            print(f"Error leyendo etiquetas {label_path}: {e}")
            return np.array([])

    def __align_keypoints(self, true_keypoints, pred_keypoints):
        """
        Alinea los keypoints verdaderos y predichos según el mapeo.
        """
        aligned_true = []
        aligned_pred = []
        
        for i, mapped_idx in enumerate(self.keypoint_mapping):
            if mapped_idx < len(true_keypoints) and i < len(pred_keypoints):
                aligned_true.append(true_keypoints[mapped_idx])
                aligned_pred.append(pred_keypoints[i])
        
        return np.array(aligned_true), np.array(aligned_pred)

    def __get_inference(self, image_path, true_keypoints, image, results, threshold):
        """
        Realiza la inferencia y calcula el PCK para una imagen.
        """
        try:
            # Realizar inferencia con MMPose
            pose_results = inference_topdown(self.model, image_path)
            data_samples = merge_data_samples(pose_results)
            
            if not data_samples.pred_instances or len(data_samples.pred_instances.keypoints) == 0:
                print(f"No se encontraron keypoints en la imagen {image}.")
                return results
            
            # Obtener keypoints predichos
            pred_keypoints = data_samples.pred_instances.keypoints[0]
            pred_scores = data_samples.pred_instances.keypoint_scores[0]
            
            print(f"Keypoints predichos: {len(pred_keypoints)}, Keypoints verdaderos: {len(true_keypoints)}")
            
            # Filtrar keypoints con baja confianza
            confidence_threshold = 0.3
            valid_indices = np.where(pred_scores > confidence_threshold)[0]
            
            if len(valid_indices) == 0:
                print(f"No hay keypoints con confianza suficiente en {image}.")
                return results
                
            pred_keypoints = pred_keypoints[valid_indices]
            
            # Normalizar keypoints verdaderos y predichos a [0,1]
            image_read = cv2.imread(image_path)
            height, width = image_read.shape[:2]
            
            true_keypoints_normalized = true_keypoints.copy()
            true_keypoints_normalized[:, 0] /= width
            true_keypoints_normalized[:, 1] /= height
            
            pred_keypoints_normalized = pred_keypoints.copy()
            pred_keypoints_normalized[:, 0] /= width
            pred_keypoints_normalized[:, 1] /= height
            
            # Alinear keypoints según el mapeo
            aligned_true, aligned_pred = self.__align_keypoints(true_keypoints_normalized, pred_keypoints_normalized)
            
            if len(aligned_true) == 0 or len(aligned_pred) == 0:
                print(f"No se pudieron alinear keypoints para {image}.")
                return results
            
            # Calcular PCK
            pck, total_true, visible_true, total_pred, visible_pred = self.__calculate_pck(
                aligned_true, aligned_pred, threshold)
            
            print(f'Image: {image} PCK: {pck:.2f}%, True: {visible_true}/{total_true}, Pred: {visible_pred}/{total_pred}')

            results.append({
                'nombre_imagen': image,
                'threshold': threshold,
                'image_size': f"{width}x{height}",
                'cantidad_true_keypoints': total_true,
                'true_keypoints_visible': visible_true,
                'cantidad_pred_keypoints': total_pred,
                'pred_keypoints_visible': visible_pred,
                'pck': pck
            })
            
        except Exception as e:
            print(f"Error en inferencia para {image}: {e}")
        
        return results

    def evaluate_image(self, image, threshold, results):
        """
        Evalúa una imagen y calcula el PCK.
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        if not self.__image_exists(image_path, label_path):
            return results
        
        try:
            # Obtener dimensiones de la imagen para normalización
            img = cv2.imread(image_path)
            if img is None:
                print(f"No se pudo cargar la imagen {image_path}")
                return results
                
            height, width = img.shape[:2]
            
            true_keypoints = self.__get_true_keypoints(label_path, width, height)
            if len(true_keypoints) == 0:
                print(f"No se pudieron obtener keypoints verdaderos para {image}")
                return results
                
            return self.__get_inference(image_path, true_keypoints, image, results, threshold)
            
        except Exception as e:
            print(f"Error evaluando {image}: {e}")
            return results

    def draw_original_keypoints(self, image_path, label_path):
        """
        Dibuja los keypoints originales en la imagen.
        
        :param image_path: Ruta de la imagen
        :param label_path: Ruta del archivo de etiquetas
        :return: Imagen con los keypoints originales dibujados
        """
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
        Dibuja los keypoints predichos en la imagen.
        
        :param image_path: Ruta de la imagen
        :return: Imagen con los keypoints predichos y resultados de inferencia
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

        # Realizar inferencia con MMPose
        pose_results = inference_topdown(self.model, image_path)
        data_samples = merge_data_samples(pose_results)
        
        if data_samples.pred_instances:
            keypoints = data_samples.pred_instances.keypoints[0]
            scores = data_samples.pred_instances.keypoint_scores[0]
            
            # Dibujar keypoints con alta confianza
            for kpt, score in zip(keypoints, scores):
                if score > 0.3:  # Umbral de confianza
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        return image, data_samples