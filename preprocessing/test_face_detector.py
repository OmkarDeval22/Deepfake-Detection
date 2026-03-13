from preprocessing.face_detector import detect_and_crop_faces

input_folder = r"F:\deepfake\dataset\frames_test"
output_folder = r"F:\deepfake\dataset\faces_test"

detect_and_crop_faces(input_folder, output_folder)

print("Face detection complete.")