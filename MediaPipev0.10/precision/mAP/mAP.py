import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

class mAP:
    def __init__(self, basepath, images_path, labels_path):
        """
        Inicializa la clase mAP con MediaPipe.
        :param basepath: Ruta base donde se encuentran los archivos.
        :param images_path: Ruta a las imágenes.
        :param labels_path: Ruta a las etiquetas.
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Keypoint mapping for 17 keypoints
        self.keypoint_mapping = {
            0: self.mp_pose.PoseLandmark.NOSE,
            1: self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
            2: self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
            3: self.mp_pose.PoseLandmark.LEFT_EAR,
            4: self.mp_pose.PoseLandmark.RIGHT_EAR,
            5: self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            6: self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            7: self.mp_pose.PoseLandmark.LEFT_ELBOW,
            8: self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            9: self.mp_pose.PoseLandmark.LEFT_WRIST,
            10: self.mp_pose.PoseLandmark.RIGHT_WRIST,
            11: self.mp_pose.PoseLandmark.LEFT_HIP,
            12: self.mp_pose.PoseLandmark.RIGHT_HIP,
            13: self.mp_pose.PoseLandmark.LEFT_KNEE,
            14: self.mp_pose.PoseLandmark.RIGHT_KNEE,
            15: self.mp_pose.PoseLandmark.LEFT_ANKLE,
            16: self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }

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
        Dibuja los keypoints predichos por MediaPipe en la imagen.
        :param image_path: Ruta de la imagen.
        :return: Imagen con los keypoints predichos dibujados y los resultados de la inferencia.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        # Draw the pose landmarks
        if results.pose_landmarks:
            # Get all landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Draw only the selected 17 keypoints
            for idx, landmark in self.keypoint_mapping.items():
                landmark_point = landmarks[landmark]
                if landmark_point.visibility > 0.5:  # MediaPipe visibility threshold
                    x = int(landmark_point.x * image.shape[1])
                    y = int(landmark_point.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            # Also draw the bounding box based on keypoints
            keypoints = self._get_predicted_keypoints(results, image.shape[1], image.shape[0])
            
            if len(keypoints) > 0:
                # Filter keypoints with visibility > 0 and convert to integers
                visible_keypoints = [(int(kp[0]), int(kp[1])) for kp in keypoints if kp[2] > 0.5]

                if visible_keypoints:  # Only proceed if we have visible keypoints
                    x_coords = [kp[0] for kp in visible_keypoints]
                    y_coords = [kp[1] for kp in visible_keypoints]
                    
                    # Get min/max coordinates
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Ensure coordinates are within image bounds
                    height, width = image.shape[:2]
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width - 1, x_max)
                    y_max = min(height - 1, y_max)
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        return image, results

    def _get_predicted_keypoints(self, results, image_width, image_height):
        """
        Extrae los keypoints predichos por MediaPipe.
        :param results: Resultados de MediaPipe.
        :param image_width: Ancho de la imagen.
        :param image_height: Alto de la imagen.
        :return: Array de keypoints (Nx3: x, y, visibility).
        """
        if not results.pose_landmarks:
            return np.zeros((17, 3))
        
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        
        for idx, landmark in self.keypoint_mapping.items():
            landmark_point = landmarks[landmark]
            x = landmark_point.x * image_width
            y = landmark_point.y * image_height
            visibility = landmark_point.visibility
            keypoints.append([x, y, visibility])
            
        return np.array(keypoints)

    def normalize_keypoints(self, keypoints, image_width, image_height):
        """
        Normaliza las coordenadas de los keypoints.
        :param keypoints: Keypoints en coordenadas absolutas (Nx3: x, y, vis).
        :param image_width: Ancho de la imagen.
        :param image_height: Alto de la imagen.
        :return: Keypoints normalizados (Nx3).
        """
        keypoints[:, 0] /= image_width
        keypoints[:, 1] /= image_height
        return keypoints

    def calculate_oks(self, true_keypoints, pred_keypoints, scale, threshold):
        """
        Calcula el OKS (Object Keypoint Similarity) de manera que responda a diferentes thresholds.
        :param true_keypoints: Keypoints verdaderos normalizados (Nx3: x, y, vis).
        :param pred_keypoints: Keypoints predichos normalizados (Nx3: x, y, vis).
        :param scale: Factor de escala (debe ser consistente).
        :param threshold: Umbral para considerar un keypoint como correcto.
        :return: OKS para cada keypoint (0 o 1).
        """
        k = len(true_keypoints)
        oks = np.zeros(k)
        
        # Asegurarse de que la escala no sea demasiado pequeña
        scale = max(scale, 0.1)  # Evitar divisiones por cero o escalas mínimas

        for i in range(k):
            x_true, y_true, vis_true = true_keypoints[i]
            x_pred, y_pred, vis_pred = pred_keypoints[i]

            if vis_true > 0 and vis_pred > 0.5:  # Solo considerar keypoints visibles
                # Calcular distancia euclidiana normalizada
                d = np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)
                # Calcular OKS y aplicar threshold
                oks[i] = 1 if (d <= threshold * scale) else 0
            else:
                oks[i] = 0  # Considerar incorrecto si no es visible

        return oks

    def calculate_ap(self, true_keypoints, pred_keypoints, scale, threshold, image, results):
        """
        Calcula el AP (Average Precision) con el nuevo cálculo de OKS.
        :param true_keypoints: Keypoints verdaderos (Nx3: x, y, vis).
        :param pred_keypoints: Keypoints predichos (Nx3: x, y, vis).
        :param scale: Factor de escala.
        :param threshold: Umbral para considerar un keypoint como correcto.
        :return: AP.
        """
        oks = self.calculate_oks(true_keypoints, pred_keypoints, scale, threshold)
        
        visible_mask = true_keypoints[:, 2] > 0
        oks = oks[visible_mask]
        
        true_positives = np.sum(oks == 1)
        false_positives = len(oks) - true_positives  # Todos los no correctos son FP
        false_negatives = np.sum(visible_mask) - true_positives  # Los visibles no detectados
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        ap = precision
        
        print(f'Imagen: {image}, threshold: {threshold:.2f}, true_keypoints: {len(true_keypoints)}, '
            f'num_visible: {np.sum(visible_mask)}, true_positives: {true_positives}, '
            f'false_positives: {false_positives}, false_negatives: {false_negatives}, '
            f'recall: {recall:.4f}, ap: {ap:.4f}')
        
        results.append({
            'image': image,
            'threshold': threshold,
            'true_keypoints': len(true_keypoints),
            'num_visible': np.sum(visible_mask),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'recall': recall,
            'ap': ap
        })
        
        return results

    def evaluate_image(self, image_name, threshold, results):
        """
        Evalúa una imagen con el nuevo cálculo de OKS.
        """
        image_path = os.path.join(self.images_path, image_name)
        label_path = os.path.join(self.labels_path, os.path.splitext(image_name)[0] + '.txt')

        # Obtener keypoints verdaderos
        true_keypoints = self.get_true_keypoints(label_path)
        
        # Obtener keypoints predichos
        img_with_predicted_keypoints, inference_results = self.draw_predicted_keypoints(image_path)
        pred_keypoints = self._get_predicted_keypoints(inference_results, 
                                                    img_with_predicted_keypoints.shape[1], 
                                                    img_with_predicted_keypoints.shape[0])
        
        # Normalizar keypoints predichos
        pred_keypoints_normalized = self.normalize_keypoints(pred_keypoints, 
                                                        img_with_predicted_keypoints.shape[1], 
                                                        img_with_predicted_keypoints.shape[0])

        # Calcular escala basada en el área de la persona (normalizada)
        visible_true = true_keypoints[true_keypoints[:, 2] > 0]
        if len(visible_true) > 0:
            x_min, x_max = np.min(visible_true[:, 0]), np.max(visible_true[:, 0])
            y_min, y_max = np.min(visible_true[:, 1]), np.max(visible_true[:, 1])
            scale = np.sqrt((x_max - x_min) * (y_max - y_min))
        else:
            scale = 0.1  # Valor por defecto
        
        # Asegurar que la escala sea significativa
        scale = max(scale, 0.05)  # No permitir escalas demasiado pequeñas

        return self.calculate_ap(true_keypoints, pred_keypoints_normalized, scale, threshold, image_name, results)
