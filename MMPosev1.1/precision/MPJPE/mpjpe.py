import os, math
import numpy as np
import pandas as pd
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

class MPJPE:
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

    def __get_true_keypoints(self, label_path):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas.
        :param label_path: Ruta del archivo de etiquetas
        :return: Coordenadas verdaderas de los keypoints
        """
        with open(label_path, 'r') as file:
            line = file.readline().strip()
            parts = line.split()

        true_keypoints = []
        for i in range(5, len(parts), 3):
            x = float(parts[i])
            y = float(parts[i+1])
            vis = int(parts[i+2])
            true_keypoints.append([x, y, vis])
        return np.array(true_keypoints)
    
    def __euclidean_distance(self, true_keypoints, pred_keypoints):
        """
        Calcula las distancias entre las coordenadas verdaderas y las predichas.
        :param true_keypoints: Coordenadas verdaderas de los keypoints
        :param pred_keypoints: Coordenadas predichas de los keypoints
        :return: Distancias entre las coordenadas verdaderas y las predichas
        """
        distances = []
        for true_kp, pred_kp in zip(true_keypoints, pred_keypoints):
            if np.any(pred_kp != 0):  # Solo considerar keypoints predichos válidos
                true_x, true_y = true_kp[0], true_kp[1]
                pred_x, pred_y = pred_kp[0], pred_kp[1]
                distance = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
                distances.append(distance)
        return distances
    
    def __calculate_mpjpe(self, true_keypoints, pred_keypoints):
        """
        Calcula el PCK dados los puntos clave verdaderos y predichos.
        :param true_keypoints: Coordenadas verdaderas de los keypoints
        :param pred_keypoints: Coordenadas predichas de los keypoints
        :return: mpjpe
        """
        visible_indices = np.where(true_keypoints[:, 2] == 2)[0]
        true_keypoints_visible = true_keypoints[visible_indices, :2]
        predicted_keypoints_visible = pred_keypoints[visible_indices, :2]
        distances = self.__euclidean_distance(true_keypoints_visible, predicted_keypoints_visible)
        mpjpe = np.mean(distances) if distances else None

        return mpjpe, true_keypoints_visible, predicted_keypoints_visible
    
    def __calculate_diff_in_pixels(self, image_width, image_height, mpjpe):
        """
        Convierte el MPJPE (en unidades normalizadas) a píxeles.
        
        :param image_width: Ancho de la imagen en píxeles.
        :param image_height: Alto de la imagen en píxeles.
        :param mpjpe: Valor del MPJPE en unidades normalizadas (0 a 1).
        :return: Distancia euclidiana en píxeles.
        """
        # Calcular la diagonal de la imagen en píxeles
        diagonal = math.sqrt(image_width**2 + image_height**2)
        
        # Convertir el MPJPE a píxeles
        mpjpe_pixels = mpjpe * diagonal
    
        return mpjpe_pixels

    def __get_inference(self, image_path, true_keypoints, image, results):
        """
        Realiza la inferencia y calcula el PCK para una imagen.
        :param image_path: Ruta de la imagen
        :param true_keypoints: Coordenadas verdaderas de los keypoints
        :param image: Nombre de la imagen
        :param results: Lista para almacenar los resultados
        """
        # Realizar inferencia top-down con MMPose
        pose_results = inference_topdown(self.model, image_path)
        data_samples = merge_data_samples(pose_results)
        
        if data_samples.pred_instances:
            pred_keypoints = data_samples.pred_instances.keypoints[0]
            image_width, image_height = data_samples.metainfo['img_shape'][1], data_samples.metainfo['img_shape'][0]
            
            # Normalizar las coordenadas predichas
            pred_keypoints[:, 0] /= image_width
            pred_keypoints[:, 1] /= image_height

            mpjpe, true_keypoints_visible, pred_keypoints_visible = self.__calculate_mpjpe(true_keypoints, pred_keypoints)
            pixels = self.__calculate_diff_in_pixels(image_width, image_height, mpjpe)

            results.append({
                'nombre_imagen': image,
                'image_size': f"{image_width}x{image_height}",
                'cantidad_true_keypoints': len(true_keypoints),
                'true_keypoints_visible': len(true_keypoints_visible),
                'cantidad_pred_keypoints': len(pred_keypoints),
                'pred_keypoints_visible': len(pred_keypoints_visible),
                'mpjpe_pixels': pixels,
                'mpjpe': mpjpe
            })

            return results
        else:
            print(f"No se encontraron keypoints en la imagen {image}.")
            return results

    def evaluate_image(self, image, results):
        """
        Evalúa una imagen y calcula el PCK.
        :param image: Nombre de la imagen
        :return: Lista de resultados
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        self.__image_exists(image_path, label_path)
        true_keypoints = self.__get_true_keypoints(label_path)

        return self.__get_inference(image_path, true_keypoints, image, results)