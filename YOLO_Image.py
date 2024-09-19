import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

def image_detection(image_path):
    """
    Perform object detection on an image using YOLOv8.
    
    Parameters:
    image_path (str): The path to the input image.
    
    Returns:
    detection_image: Image with detected objects and bounding boxes.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Perform detection using the YOLO model
    results = model(image)
    
    # Extract the results and draw bounding boxes
    annotated_image = results[0].plot()  # This will draw bounding boxes and labels on the image
    
    return annotated_image
