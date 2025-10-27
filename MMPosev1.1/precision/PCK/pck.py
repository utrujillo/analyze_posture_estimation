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
        
        :param basepath: Ruta base del proyecto
        :param images_path: Ruta de las imágenes
        :param labels_path: Ruta de las etiquetas
        :param config_file: Archivo de configuración del modelo MMPose
        :param checkpoint_file: Checkpoint del modelo MMPose
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        
        # Inicializar el modelo MMPose
        self.model = init_model(config_file, checkpoint_file, device='cpu')
        
    def __calculate_pck(self, true_keypoints, predicted_keypoints, threshold):
        """
        Calcula el PCK dados los puntos clave verdaderos y predichos.
        
        :param true_keypoints: Coordenadas verdaderas de los keypoints (N x 2)
        :param predicted_keypoints: Coordenadas predichas de los keypoints (N x 2)
        :param threshold: Umbral de distancia para considerar un punto clave correcto
        :return: PCK y estadísticas
        """
        visible_indices = np.where(true_keypoints[:, 2] == 2)[0]
        true_keypoints_visible = true_keypoints[visible_indices, :2]
        predicted_keypoints_visible = predicted_keypoints[visible_indices, :2]
        
        distances = np.linalg.norm(true_keypoints_visible - predicted_keypoints_visible, axis=1)
        correct = np.sum(distances < threshold)
        total = len(true_keypoints_visible)
        
        pck = (correct / total) * 100 if total > 0 else 0.0
        return pck, len(true_keypoints), len(true_keypoints_visible), len(predicted_keypoints), len(predicted_keypoints_visible)

    def __image_exists(self, image_path, label_path):
        """
        Verifica si la imagen y el archivo de etiquetas existen.
        
        :param image_path: Ruta de la imagen
        :param label_path: Ruta del archivo de etiquetas
        """
        if not os.path.exists(image_path):
            print(f"La imagen {image_path} no existe.")
        if not os.path.exists(label_path):
            print(f"El archivo de etiquetas {label_path} no existe.")

    def __get_true_keypoints(self, label_path, image_width, image_height):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas.
        
        :param label_path: Ruta del archivo de etiquetas
        :param image_width: Ancho de la imagen
        :param image_height: Alto de la imagen
        :return: Coordenadas verdaderas de los keypoints
        """
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

    def __get_inference(self, image_path, true_keypoints, image, results, threshold):
        """
        Realiza la inferencia y calcula el PCK para una imagen.
        
        :param image_path: Ruta de la imagen
        :param true_keypoints: Coordenadas verdaderas de los keypoints
        :param image: Nombre de la imagen
        :param results: Lista para almacenar los resultados
        :param threshold: Umbral de distancia para considerar un punto clave correcto
        """
        # Realizar inferencia con MMPose
        pose_results = inference_topdown(self.model, image_path)
        data_samples = merge_data_samples(pose_results)
        
        if data_samples.pred_instances:
            # Obtener keypoints predichos
            pred_keypoints = data_samples.pred_instances.keypoints[0]
            pred_scores = data_samples.pred_instances.keypoint_scores[0]
            
            # Filtrar keypoints con baja confianza
            confidence_threshold = 0.3
            valid_indices = np.where(pred_scores > confidence_threshold)[0]
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
            
            # Calcular PCK
            pck, total_true, visible_true, total_pred, visible_pred = self.__calculate_pck(
                true_keypoints_normalized, pred_keypoints_normalized, threshold)
            
            print(f'PCK: {pck:.2f}%, Total true: {total_true}, Visible true: {visible_true}, '
                  f'Total pred: {total_pred}, Visible pred: {visible_pred}')

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
        else:
            print(f"No se encontraron keypoints en la imagen {image}.")
        
        return results

    def evaluate_image(self, image, threshold, results):
        """
        Evalúa una imagen y calcula el PCK.
        
        :param image: Nombre de la imagen
        :param threshold: Umbral de distancia para considerar un punto clave correcto
        :param results: Lista de resultados
        :return: Lista de resultados actualizada
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        self.__image_exists(image_path, label_path)
        
        # Obtener dimensiones de la imagen para normalización
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        true_keypoints = self.__get_true_keypoints(label_path, width, height)
        return self.__get_inference(image_path, true_keypoints, image, results, threshold)

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