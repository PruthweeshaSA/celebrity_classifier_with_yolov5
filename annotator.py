import os
from facenet_pytorch import MTCNN
import cv2
from tqdm import tqdm
import torch
import numpy as np

# Paths
dataset_root = 'CelebrityFacesDataset_Curated'
output_root = 'yolo_dataset'
image_output_dir = os.path.join(output_root, 'images')
label_output_dir = os.path.join(output_root, 'labels')
visual_output_dir = os.path.join(output_root, 'box_visualizations')

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)
os.makedirs(visual_output_dir, exist_ok=True)

# Init face detector
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

no_face_virtual_label = '__No Face'

def resize_with_padding(img, target_size=512, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image with preserved aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a new square image and paste resized image into center
    padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

# Build class index
celebrity_names = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('_')])
celebrity_names = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith('_')], reverse=True) + celebrity_names
celebrity_names.remove(no_face_virtual_label)
class_map = {name: idx for idx, name in enumerate(celebrity_names)}

# Process images
for celeb_name in tqdm(celebrity_names):
    class_id = class_map[celeb_name]
    celeb_dir = os.path.join(dataset_root, celeb_name)

    for img_name in os.listdir(celeb_dir):
        img_path = os.path.join(celeb_dir, img_name)
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = resize_with_padding(img, target_size=512)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is None:
            continue

        boxes = sorted(boxes, key=(lambda x: x[2]-x[0]), reverse=True )

        for i, box in enumerate(boxes):
            if box is None or len(box) != 4:
                continue

            x1, y1, x2, y2 = box
            h, w = img.shape[:2]

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            # Reject invalid boxes
            if x2 <= x1 or y2 <= y1:
                print(f"Skipped corrupt box: {box}")
                continue

            # Normalize for YOLO
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            # Reject extremely small boxes
            if width <= 0 or height <= 0 or width > 1 or height > 1:
                print(f"Skipped corrupt normalized box: {width=:.4f}, {height=:.4f}")
                continue

            # Save image to common output
            output_img_path = os.path.join(image_output_dir, f'{celeb_name}_{img_name}')
            cv2.imwrite(output_img_path, img)

            # Save label
            label_path = os.path.join(label_output_dir, f'{celeb_name}_{img_name}'.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Draw bounding box on a copy and save
            vis_img = img.copy()
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
            vis_path = os.path.join(visual_output_dir, f'{celeb_name}_{img_name}')
            cv2.imwrite(vis_path, vis_img)

            break


## Add no face directory
celeb_name = f"{no_face_virtual_label}"
celeb_dir = os.path.join(dataset_root, celeb_name)
for img_name in os.listdir(celeb_dir):
    img_path = os.path.join(celeb_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = resize_with_padding(img, target_size=512)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save image to common output
    output_img_path = os.path.join(image_output_dir, f'{celeb_name}_{img_name}')
    cv2.imwrite(output_img_path, img)

    # Save label
    label_path = os.path.join(label_output_dir, f'{celeb_name}_{img_name}'.replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(label_path, 'w') as f:
        f.write(f"\n")