import os, cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class PCK:
    def __init__(self, basepath, images_path, labels_path, model_name='yolo11n-pose.pt'):
        self.basepath = basepath
        self.model_path = os.path.join(basepath, model_name)
        self.model = YOLO(self.model_path)
        self.images_path = images_path
        self.labels_path = labels_path

    def __calculate_pck(self, true_keypoints, predicted_keypoints, threshold):
        """
        Calcula el PCK dados los puntos clave verdaderos y predichos.
        :param true_keypoints: Coordenadas verdaderas de los keypoints (N x 2)
        :param predicted_keypoints: Coordenadas predichas de los keypoints (N x 2)
        :param threshold: Umbral de distancia para considerar un punto clave correcto
        :return: PCK
        """
        visible_indices = np.where(true_keypoints[:, 2] == 2)[0]
        true_keypoints_visible = true_keypoints[visible_indices, :2]
        predicted_keypoints_visible = predicted_keypoints[visible_indices, :2]
        
        distances = np.linalg.norm(true_keypoints_visible - predicted_keypoints_visible, axis=1)
        correct = np.sum(distances < threshold)
        total = len(true_keypoints_visible)
        
        pck = (correct / total) * 100
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

    def __get_inference(self, image_path, true_keypoints, image, results, threshold):
        """
        Realiza la inferencia y calcula el PCK para una imagen.
        :param image_path: Ruta de la imagen
        :param true_keypoints: Coordenadas verdaderas de los keypoints
        :param image: Nombre de la imagen
        :param results: Lista para almacenar los resultados
        :param threshold: Umbral de distancia para considerar un punto clave correcto
        """
        results_pred = self.model.predict(source=image_path, save=True, save_txt=False)

        for result in results_pred:
            if result.keypoints is not None:
                pred_keypoints = result.keypoints.data[0].cpu().numpy()
                pred_keypoints = pred_keypoints[:, :2]
                image_width, image_height = result.orig_shape[1], result.orig_shape[0]
                pred_keypoints[:, 0] /= image_width
                pred_keypoints[:, 1] /= image_height

                pck, total_true, visible_true, total_pred, visible_pred = self.__calculate_pck(true_keypoints, pred_keypoints, threshold)
                print(f'pck: {pck}, total_true: {total_true}, visible_true: {visible_true}, total_pred: {total_pred}, visible_pred: {visible_pred}')

                results.append({
                    'nombre_imagen': image,
                    'threshold': threshold,
                    'image_size': f"{image_width}x{image_height}",
                    'cantidad_true_keypoints': total_true,
                    'true_keypoints_visible': visible_true,
                    'cantidad_pred_keypoints': total_pred,
                    'pred_keypoints_visible': visible_pred,
                    'pck': pck
                })

                return results
            else:
                print(f"No se encontraron keypoints en la imagen {image}.")
        
        return results

    def evaluate_image(self, image, threshold, results):
        """
        Evalúa una imagen y calcula el PCK.
        :param image: Nombre de la imagen
        :param threshold: Umbral de distancia para considerar un punto clave correcto
        :return: Lista de resultados
        """
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')

        self.__image_exists(image_path, label_path)
        true_keypoints = self.__get_true_keypoints(label_path)
        return self.__get_inference(image_path, true_keypoints, image, results, threshold)

    def draw_original_keypoints(self, image_path, label_path):
        """
        Dibuja los keypoints originales en la imagen.
        :param image_path: Ruta de la imagen.
        :param label_path: Ruta del archivo de etiquetas.
        :return: Imagen con los keypoints originales dibujados.
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