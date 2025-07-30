import yaml
import argparse
import cv2
import numpy as np
import torch
from src.pipelines.gaze_pipeline import GazePipeline

def draw_gaze(image, bbox, gaze_data, color=(0, 0, 255)):
    """Draws the gaze vector on the image."""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    pitch = np.radians(gaze_data['pitch'])
    yaw = np.radians(gaze_data['yaw'])

    length = 150 # Length of the gaze vector
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)

    end_point = (int(center_x + dx), int(center_y + dy))

    cv2.arrowedLine(image, (center_x, center_y), end_point, color, 2, tipLength=0.2)
    return image

def main(cfg_path, weights_path, source):
    # --- 1. Load Config and Initialize Pipeline ---
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = GazePipeline(cfg, device, weights_path)

    # --- 2. Setup Video Capture ---
    try:
        video_source = int(source)
    except ValueError:
        video_source = source

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return

    # --- 3. Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with the pipeline
        results = pipeline(frame)

        # Draw results on the frame
        for res in results:
            bbox = res['bbox']
            gaze = res['gaze']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            frame = draw_gaze(frame, bbox, gaze)

        cv2.imshow('Gaze Estimation Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze Estimation Demo")
    parser.add_argument('--config', type=str, required=True, help="Path to the training config file.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument('--source', type=str, default='0', help="Video source (path to file or camera ID).")
    args = parser.parse_args()
    main(args.config, args.weights, args.source)
