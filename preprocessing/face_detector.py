import torch
import cv2
import os
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)


def detect_and_crop_faces(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face = mtcnn(image_rgb)

        if face is not None:
            face_img = face.permute(1, 2, 0).cpu().numpy()
            face_img = (face_img * 255).astype("uint8")
            cv2.imwrite(os.path.join(output_folder, img_name), face_img)