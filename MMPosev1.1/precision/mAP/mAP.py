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
        """
        self.basepath = basepath
        self.images_path = images_path
        self.labels_path = labels_path
        
        # Inicializar el modelo MMPose
        self.model = init_model(config_file, checkpoint_file, device='cpu')
        
        # Mapeo de keypoints - AJUSTA ESTO SEGÚN TU MODELO MMPose
        self.keypoint_mapping = self._get_keypoint_mapping()

    def _get_keypoint_mapping(self):
        """
        Define el mapeo entre tus 17 keypoints y los keypoints del modelo MMPose.
        Esto es un mapeo genérico - AJUSTA según tu modelo específico.
        """
        # Mapeo por defecto (asume mismo orden)
        return list(range(17))

    def get_true_keypoints(self, label_path):
        """
        Obtiene las coordenadas verdaderas de los keypoints desde el archivo de etiquetas.
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

    def align_keypoints(self, true_keypoints, pred_keypoints):
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

    def draw_original_keypoints(self, image_path, label_path):
        """
        Dibuja los keypoints originales en la imagen.
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
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

        # Realizar inferencia con MMPose
        pose_results = inference_topdown(self.model, image_path)
        data_samples = merge_data_samples(pose_results)
        
        if not data_samples.pred_instances or len(data_samples.pred_instances.keypoints) == 0:
            print("No se detectaron personas en la imagen")
            return image, data_samples
        
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
        """
        keypoints_normalized = keypoints.copy()
        keypoints_normalized[:, 0] /= image_width
        keypoints_normalized[:, 1] /= image_height
        return keypoints_normalized

    def calculate_oks(self, true_keypoints, pred_keypoints, scale):
        """
        Calcula el OKS (Object Keypoint Similarity).
        """
        k = min(len(true_keypoints), len(pred_keypoints))  # Usar el mínimo de keypoints
        oks = np.zeros(k)

        for i in range(k):
            x_true, y_true, vis_true = true_keypoints[i]
            x_pred, y_pred = pred_keypoints[i]

            if vis_true > 0:  # Solo considerar keypoints visibles
                d = (x_true - x_pred) ** 2 + (y_true - y_pred) ** 2
                oks[i] = np.exp(-d / (2 * scale ** 2 + 1e-7))  # Evitar división por cero

        return oks

    def calculate_ap(self, true_keypoints, pred_keypoints, scale, threshold, image, results):
        """
        Calcula el AP (Average Precision).
        """
        # Alinear keypoints antes de calcular OKS
        aligned_true, aligned_pred = self.align_keypoints(true_keypoints, pred_keypoints)
        
        if len(aligned_true) == 0 or len(aligned_pred) == 0:
            print(f"No se pudieron alinear keypoints para {image}")
            return results

        oks = self.calculate_oks(aligned_true, aligned_pred, scale)

        visible_mask = aligned_true[:, 2] > 0
        oks_visible = oks[visible_mask]

        if len(oks_visible) == 0:
            print(f"No hay keypoints visibles para calcular AP en {image}")
            return results

        correct = oks_visible >= threshold
        num_visible = np.sum(visible_mask)
        true_positives = np.sum(correct)
        false_positives = len(oks_visible) - true_positives
        false_negatives = num_visible - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        ap = precision
        
        print(f'Imagen: {image}, threshold: {threshold}')
        print(f'  Keypoints - True: {len(true_keypoints)}, Pred: {len(pred_keypoints)}, Alineados: {len(aligned_true)}')
        print(f'  Visible: {num_visible}, TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}')
        print(f'  Precision: {precision:.3f}, Recall: {recall:.3f}, AP: {ap:.3f}')
        
        results.append({
            'image': image,
            'threshold': threshold,
            'true_keypoints': len(true_keypoints),
            'pred_keypoints': len(pred_keypoints),
            'aligned_keypoints': len(aligned_true),
            'num_visible': num_visible,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'ap': ap
        })

        return results

    def evaluate_image(self, image_name, threshold, results):
        """
        Evalúa una imagen y calcula el AP.
        """
        image_path = os.path.join(self.images_path, image_name)
        label_path = os.path.join(self.labels_path, os.path.splitext(image_name)[0] + '.txt')

        # Verificar que los archivos existen
        if not os.path.exists(image_path):
            print(f"La imagen {image_path} no existe")
            return results
        if not os.path.exists(label_path):
            print(f"El archivo de etiquetas {label_path} no existe")
            return results

        try:
            # Obtener keypoints verdaderos
            true_keypoints = self.get_true_keypoints(label_path)
            
            if len(true_keypoints) == 0:
                print(f"No se pudieron obtener keypoints verdaderos para {image_name}")
                return results

            # Obtener keypoints predichos con MMPose
            img_with_predicted_keypoints, inference_results = self.draw_predicted_keypoints(image_path)
            
            if not inference_results.pred_instances or len(inference_results.pred_instances.keypoints) == 0:
                print(f"No se detectaron personas en {image_name}")
                return results
            
            # Obtener keypoints predichos (tomamos solo la primera persona detectada)
            pred_keypoints = inference_results.pred_instances.keypoints[0]
            scores = inference_results.pred_instances.keypoint_scores[0]
            
            # Filtrar keypoints con baja confianza
            valid_indices = scores > 0.3
            pred_keypoints = pred_keypoints[valid_indices]
            
            if len(pred_keypoints) == 0:
                print(f"No hay keypoints con confianza suficiente en {image_name}")
                return results
            
            # Normalizar keypoints predichos
            image_height, image_width, _ = img_with_predicted_keypoints.shape
            pred_keypoints_normalized = self.normalize_keypoints(pred_keypoints, image_width, image_height)

            # Calcular la escala (usamos el área del bounding box predicho)
            bbox = inference_results.pred_instances.bboxes[0]
            scale = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / image_width
            scale = max(scale / 10, 0.01)  # Ajustar la escala con mínimo

            # Calcular el AP
            return self.calculate_ap(true_keypoints, pred_keypoints_normalized, scale, threshold, image_name, results)
            
        except Exception as e:
            print(f"Error evaluando {image_name}: {e}")
            return results