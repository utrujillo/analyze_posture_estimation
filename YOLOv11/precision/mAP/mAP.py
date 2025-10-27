import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


class mAP:
    def __init__(self, basepath, images_path, labels_path, model_name='yolo11n-pose.pt'):
        """
        Inicializa la clase mAP.
        :param basepath: Ruta base donde se encuentran los archivos.
        :param model_name: Nombre del archivo del modelo YOLO.
        """
        self.basepath = basepath
        self.model_path = os.path.join(basepath, model_name)
        self.model = YOLO(self.model_path)
        self.images_path = images_path
        self.labels_path = os.path.join(basepath, 'dataset/labels/')
        # self.labels_path = os.path.join(basepath, 'manually_keypoints/labels/')

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
        Dibuja los keypoints predichos en la imagen.
        :param image_path: Ruta de la imagen.
        :return: Imagen con los keypoints predichos dibujados y los resultados de la inferencia.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

        inference_results = self.model(image_path)

        for result in inference_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()

            for box, kpts in zip(boxes, keypoints):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                for kpt in kpts:
                    x, y = map(int, kpt)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        return image, inference_results

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

        # Obtener keypoints predichos
        img_with_predicted_keypoints, inference_results = self.draw_predicted_keypoints(image_path)
        pred_keypoints = inference_results[0].keypoints.xy.cpu().numpy().squeeze()
        
        # Verificar la forma de pred_keypoints
        if pred_keypoints.ndim == 3:
            # Si hay múltiples personas, seleccionar la primera
            pred_keypoints = pred_keypoints[0]

        # Normalizar keypoints predichos
        image_height, image_width, _ = img_with_predicted_keypoints.shape
        pred_keypoints_normalized = self.normalize_keypoints(pred_keypoints, image_width, image_height)

        # Calcular la escala
        scale = np.sqrt(inference_results[0].boxes.xywh[0][2] * inference_results[0].boxes.xywh[0][3]) / image_width
        scale = scale / 10  # Ajustar la escala

        # Calcular el AP
        return self.calculate_ap(true_keypoints, pred_keypoints_normalized, scale, threshold, image_name, results)