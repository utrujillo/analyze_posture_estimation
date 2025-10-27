# Contenido del archivo: test.py

import torch
import os
# Eliminamos 'requests' ya que no descargaremos de URL
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

# --- Configuración de Rutas y Dispositivo ---
device = 'cpu'
RTDETR_LOCAL_PATH = "./configs/detector/rtdetr_detector"
VITPOSE_LOCAL_PATH = "./configs/pose_estimation/vitpose_estimator"

# 💡 MODIFICACIONES PARA CARGA LOCAL DE IMAGEN
IMAGE_DIR = "/Users/apple/Documents/pose_estimation/dataset/images"
IMAGE_FILENAME = "imagen_008_jpg.rf.00700f922d37da70606fa16130a5d9a7.jpg"
IMAGE_PATH = os.path.join(IMAGE_DIR, IMAGE_FILENAME)


# --- Cargar Imagen Localmente ---
try:
    # Cargar la imagen directamente desde la ruta local
    image = Image.open(IMAGE_PATH)
    print(f"Imagen cargada exitosamente desde: {IMAGE_PATH}")
except FileNotFoundError:
    print(f"ERROR: No se encontró la imagen en la ruta: {IMAGE_PATH}")
    exit()

# --- 1. Detección de Personas (Carga Local de Modelos) ---
person_image_processor = AutoProcessor.from_pretrained(RTDETR_LOCAL_PATH, use_fast=True)
person_model = RTDetrForObjectDetection.from_pretrained(RTDETR_LOCAL_PATH, device_map=device)

inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)
result = results[0]

# Filtrar y formatear las cajas de las personas
person_boxes = result["boxes"][result["labels"] == 0]
person_boxes = person_boxes.cpu().numpy()

# Convertir cajas de VOC (x1, y1, x2, y2) a COCO (x1, y1, w, h)
person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

# --- 2. Estimación de Pose (Carga Local de Modelos) ---
image_processor = AutoProcessor.from_pretrained(VITPOSE_LOCAL_PATH, use_fast=True)
model = VitPoseForPoseEstimation.from_pretrained(VITPOSE_LOCAL_PATH, device_map=device)

inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
image_pose_result = pose_results[0]

xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

key_points = sv.KeyPoints(
    xy=xy, confidence=scores
)

# --- 3. Anotación y Visualización ---
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=1
)
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.RED,
    radius=2
)
annotated_frame = edge_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
)
annotated_frame = vertex_annotator.annotate(
    scene=annotated_frame,
    key_points=key_points
)

# Mostrar la imagen resultante
annotated_frame.show()