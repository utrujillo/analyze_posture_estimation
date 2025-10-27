import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples


class mAP:
    def __init__(self, basepath, images_path, labels_path, config_file, checkpoint_file):
        """
        Inicializa la clase mAP para MMPose.
        :param basepath: Ruta base donde se encuentran los archivos.
        :param config_file: Ruta al archivo de configuración del modelo MMPose.
        :param checkpoint_file: Ruta al archivo de checkpoint del modelo.
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        
        # Inicializar el modelo MMPose
        self.model = init_model(config_file, checkpoint_file, device='cpu')

    def get_true_keypoints(self, label_path):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas.
        :param label_path: Ruta del archivo de etiquetas.
        :return: Coordenadas verdaderas de los keypoints (Nx3: x, y, vis).
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
        Dibuja los keypoints originales en la imagen.
        :param image_path: Ruta de la imagen.
        :param label_path: Ruta del archivo de etiquetas.
        :return: Imagen con los keypoints originales dibujados.
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
        Dibuja los keypoints predichos en la imagen usando MMPose.
        :param image_path: Ruta de la imagen.
        :return: Imagen con los keypoints predichos dibujados y los resultados de la inferencia.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

        # Realizar inferencia con MMPose
        pose_results = inference_topdown(self.model, image_path)
        data_samples = merge_data_samples(pose_results)
        
        # Dibujar resultados
        pred_keypoints = data_samples.pred_instances.keypoints[0]  # Tomamos la primera persona detectada
        scores = data_samples.pred_instances.keypoint_scores[0]
        
        # Dibujar keypoints
        for i, (kpt, score) in enumerate(zip(pred_keypoints, scores)):
            if score > 0.3:  # Umbral de confianza
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                
        return image, data_samples

    def normalize_keypoints(self, keypoints, image_width, image_height):
        """
        Normaliza las coordenadas de los keypoints.
        :param keypoints: Keypoints en coordenadas absolutas (Nx2).
        :param image_width: Ancho de la imagen.
        :param image_height: Alto de la imagen.
        :return: Keypoints normalizados (Nx2).
        """
        keypoints[:, 0] /= image_width
        keypoints[:, 1] /= image_height
        return keypoints

    def calculate_oks(self, true_keypoints, pred_keypoints, scale):
        """
        Calcula el OKS (Object Keypoint Similarity).
        :param true_keypoints: Keypoints verdaderos (Nx3: x, y, vis).
        :param pred_keypoints: Keypoints predichos (Nx2: x, y).
        :param scale: Factor de escala.
        :return: OKS para cada keypoint.
        """
        k = len(true_keypoints)
        oks = np.zeros(k)

        for i in range(k):
            x_true, y_true, vis_true = true_keypoints[i]
            x_pred, y_pred = pred_keypoints[i]

            if vis_true > 0 and (x_pred != 0 or y_pred != 0):
                d = (x_true - x_pred) ** 2 + (y_true - y_pred) ** 2
                oks[i] = np.exp(-d / (2 * scale ** 2))

        return oks

    def calculate_ap(self, true_keypoints, pred_keypoints, scale, threshold, image, results):
        """
        Calcula el AP (Average Precision).
        :param true_keypoints: Keypoints verdaderos (Nx3: x, y, vis).
        :param pred_keypoints: Keypoints predichos (Nx2: x, y).
        :param scale: Factor de escala.
        :param threshold: Umbral para considerar un keypoint como correcto.
        :return: AP.
        """
        oks = self.calculate_oks(true_keypoints, pred_keypoints, scale)

        visible_mask = true_keypoints[:, 2] > 0
        oks = oks[visible_mask]

        correct = oks >= threshold
        num_visible = np.sum(visible_mask)
        true_positives = np.sum(correct)
        false_positives = len(correct) - true_positives
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
        Evalúa una imagen y calcula el AP.
        :param image_name: Nombre de la imagen.
        :param threshold: Umbral para considerar un keypoint como correcto.
        :return: AP.
        """
        image_path = os.path.join(self.images_path, image_name)
        label_path = os.path.join(self.labels_path, os.path.splitext(image_name)[0] + '.txt')

        # Obtener keypoints verdaderos
        true_keypoints = self.get_true_keypoints(label_path)

        # Obtener keypoints predichos con MMPose
        img_with_predicted_keypoints, inference_results = self.draw_predicted_keypoints(image_path)
        
        # Obtener keypoints predichos (tomamos solo la primera persona detectada)
        pred_keypoints = inference_results.pred_instances.keypoints[0]
        scores = inference_results.pred_instances.keypoint_scores[0]
        
        # Filtrar keypoints con baja confianza
        valid_indices = scores > 0.3
        pred_keypoints = pred_keypoints[valid_indices]
        
        # Normalizar keypoints predichos
        image_height, image_width, _ = img_with_predicted_keypoints.shape
        pred_keypoints_normalized = self.normalize_keypoints(pred_keypoints, image_width, image_height)

        # Calcular la escala (usamos el área del bounding box predicho)
        bbox = inference_results.pred_instances.bboxes[0]
        scale = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / image_width
        scale = scale / 10  # Ajustar la escala

        # Calcular el AP
        return self.calculate_ap(true_keypoints, pred_keypoints_normalized, scale, threshold, image_name, results)