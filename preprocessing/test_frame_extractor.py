from preprocessing.frame_extractor import extract_frames

video_path = r"F:\deepfake\dataset\test_video.mp4"
output_folder = r"F:\deepfake\dataset\frames_test"

extract_frames(video_path, output_folder, frame_skip=5)