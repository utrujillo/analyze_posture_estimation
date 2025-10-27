import os
import numpy as np
import pandas as pd
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

class DropRate:
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
    
    def get_true_keypoints(self, label_path):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas.
        :param label_path: Ruta del archivo de etiquetas.
        :return: Coordenadas verdaderas de los keypoints.
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
    
    def get_pred_keypoints(self, image_path):
        """
        Obtiene las coordenadas predichas de los keypoints para una imagen usando MMPose.
        :param image_path: Ruta de la imagen.
        :return: Coordenadas predichas de los keypoints (normalizadas), ancho y alto de la imagen.
        """
        # Realizar inferencia con MMPose
        results = inference_topdown(self.model, image_path)
        results = merge_data_samples(results)
        
        if results.pred_instances:
            pred_keypoints = results.pred_instances.keypoints[0]
            pred_scores = results.pred_instances.keypoint_scores[0]
            
            # Obtener dimensiones de la imagen
            img_height, img_width = results.metainfo['img_shape'][:2]
            
            # Normalizar coordenadas
            pred_keypoints[:, 0] /= img_width
            pred_keypoints[:, 1] /= img_height
            
            # Combinar coordenadas con scores (usaremos scores como visibilidad)
            pred_keypoints_with_vis = np.hstack([
                pred_keypoints, 
                pred_scores.reshape(-1, 1)
            ])
            
            return pred_keypoints_with_vis, img_width, img_height
        else:
            print(f"No se encontraron keypoints en la imagen {image_path}.")
            return None, None, None
    
    def get_drop_rate(self, true_keypoints, pred_keypoints, threshold):
        """
        Calcula el drop rate comparando los keypoints reales con los predichos.
        :param true_keypoints: Keypoints reales (etiquetados manualmente).
        :param pred_keypoints: Keypoints predichos normalizados.
        :param threshold: Umbral de distancia para considerar un keypoint como detectado.
        :return: Drop rate (porcentaje de keypoints no detectados).
        """
        # Filtrar keypoints visibles (estado == 2)
        visible_keypoints = true_keypoints[true_keypoints[:, 2] == 2]
        total_visible = len(visible_keypoints)

        # Contador de keypoints detectados correctamente
        detected_count = 0

        # Comparar keypoints uno a uno (asumiendo que están alineados por posición)
        for i, true_kp in enumerate(visible_keypoints):
            true_x, true_y = true_kp[0], true_kp[1]
            pred_x, pred_y = pred_keypoints[i, 0], pred_keypoints[i, 1]  # Keypoint predicho correspondiente

            # Calcular la distancia euclidiana
            distance = np.linalg.norm([true_x - pred_x, true_y - pred_y])
            
            # Verificar si la distancia es menor que el umbral
            if distance < threshold:
                detected_count += 1

        # Calcular el drop rate
        drop_rate = ((total_visible - detected_count) / total_visible) * 100
        return drop_rate, total_visible, detected_count

    def evaluate_image(self, image, threshold, results):
        """
        Evalúa una imagen y calcula el drop rate.
        :param image: Nombre de la imagen
        :param threshold: Umbral de distancia para considerar un punto clave correcto
        :return: Lista de resultados
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        self.__image_exists(image_path, label_path)
        true_keypoints = self.get_true_keypoints(label_path)
        pred_keypoints, image_width, image_height = self.get_pred_keypoints(image_path)
        
        if pred_keypoints is not None and image_width is not None:
            drop_rate, total_visible, detected_count = self.get_drop_rate(true_keypoints, pred_keypoints, threshold)
            
            results.append({
                'image_name': image,
                'image_size': f"{image_width}x{image_height}",
                'threshold': threshold,
                'true_keypoints': len(true_keypoints),
                'pred_keypoints': len(pred_keypoints),
                'total_visible': total_visible,
                'detected_count': detected_count,
                'drop_rate': drop_rate
            })
        
        return results