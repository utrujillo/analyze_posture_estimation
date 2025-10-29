import os
import cv2
import numpy as np
import mediapipe as mp

class PCK:
    def __init__(self, basepath, images_path, labels_path, threshold=0.5):
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        self.min_detection_confidence = threshold
        
        # Mapeo de tus 17 keypoints a los landmarks de MediaPipe
        self.keypoint_mapping = {
            0: mp.solutions.pose.PoseLandmark.NOSE,           # Nose
            1: mp.solutions.pose.PoseLandmark.LEFT_EYE,       # Left-eye
            2: mp.solutions.pose.PoseLandmark.RIGHT_EYE,      # Right-eye
            3: mp.solutions.pose.PoseLandmark.LEFT_EAR,       # Left-ear
            4: mp.solutions.pose.PoseLandmark.RIGHT_EAR,      # Right-ear
            5: mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,  # Left-shoulder
            6: mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, # Right-shoulder
            7: mp.solutions.pose.PoseLandmark.LEFT_ELBOW,     # Left-elbow
            8: mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,    # Right-elbow
            9: mp.solutions.pose.PoseLandmark.LEFT_WRIST,     # Left-wrist
            10: mp.solutions.pose.PoseLandmark.RIGHT_WRIST,   # Right-wrist
            11: mp.solutions.pose.PoseLandmark.LEFT_HIP,      # Left-hip
            12: mp.solutions.pose.PoseLandmark.RIGHT_HIP,     # Right-hip
            13: mp.solutions.pose.PoseLandmark.LEFT_KNEE,     # Left-knee
            14: mp.solutions.pose.PoseLandmark.RIGHT_KNEE,    # Right-knee
            15: mp.solutions.pose.PoseLandmark.LEFT_ANKLE,    # Left-ankle
            16: mp.solutions.pose.PoseLandmark.RIGHT_ANKLE    # Right-ankle
        }
        
        self.mp_pose = mp.solutions.pose
        self.pose = None
    
    def __initialize_pose_model(self, threshold):
        """
        Inicializa el modelo de pose de MediaPipe con el umbral de confianza especificado.
        :param threshold: Umbral de confianza para la detección de poses.
        """
        threshold = threshold if threshold is not None else self.min_detection_confidence

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=threshold,
            model_complexity=1  # 1 para modelo ligero
        )

    def __calculate_pck(self, true_keypoints, predicted_keypoints, threshold):
        visible_indices = np.where(true_keypoints[:, 2] == 2)[0]
        true_keypoints_visible = true_keypoints[visible_indices, :2]
        predicted_keypoints_visible = predicted_keypoints[visible_indices, :2]
        
        distances = np.linalg.norm(true_keypoints_visible - predicted_keypoints_visible, axis=1)
        correct = np.sum(distances < threshold)
        total = len(true_keypoints_visible)
        pck = (correct / total) * 100 if total > 0 else 0
        
        return pck, len(true_keypoints), len(true_keypoints_visible), len(predicted_keypoints), len(predicted_keypoints_visible)

    def __image_exists(self, image_path, label_path):
        if not os.path.exists(image_path):
            print(f"La imagen {image_path} no existe.")
        if not os.path.exists(label_path):
            print(f"El archivo de etiquetas {label_path} no existe.")

    def __get_true_keypoints(self, label_path):
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
        img = cv2.imread(image_path)
        if img is None:
            print(f"No se pudo leer la imagen {image_path}")
            return results
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(img_rgb)
        
        if result.pose_landmarks:
            image_height, image_width, _ = img.shape
            pred_keypoints = []
            
            # Extraer solo los 17 keypoints mapeados
            for i in range(17):
                landmark = result.pose_landmarks.landmark[self.keypoint_mapping[i]]
                pred_keypoints.append([landmark.x, landmark.y])
            
            pred_keypoints = np.array(pred_keypoints)
            
            # Calcular PCK
            pck, total_true, visible_true, total_pred, visible_pred = self.__calculate_pck(
                true_keypoints, pred_keypoints, threshold
            )
            
            print(f'PCK: {pck:.2f}%, True: {visible_true}/{total_true}, Pred: {visible_pred}/{total_pred}')
            
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
        else:
            print(f"No se encontraron keypoints en la imagen {image}.")
        
        return results

    def evaluate_image(self, image, threshold, results):
        image_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, os.path.splitext(image)[0] + '.txt')
        
        self.__initialize_pose_model(threshold)
        self.__image_exists(image_path, label_path)
        true_keypoints = self.__get_true_keypoints(label_path)
        return self.__get_inference(image_path, true_keypoints, image, results, threshold)

    def draw_predicted_keypoints(self, image_path, label_path):
        self.__image_exists(image_path, label_path)
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image}")
            
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.__initialize_pose_model(threshold=0.5)
        result = self.pose.process(img_rgb)
        
        if result.pose_landmarks:
            # Dibujar solo los 17 keypoints mapeados
            for i in range(17):
                landmark = result.pose_landmarks.landmark[self.keypoint_mapping[i]]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                # cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return image, result
    
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