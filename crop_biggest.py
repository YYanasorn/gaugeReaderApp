import cv2
import numpy as np
import torch
from ultralytics import YOLO

def crop_largest_segment(image_path, model_path, output_path):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Perform segmentation using YOLO
    results = model(image)
    
    # Extract segmentation masks and find the largest one
    max_area = 0
    largest_bbox = None
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_bbox = (x1, y1, x2, y2)
    
    if largest_bbox is None:
        print("No segment found!")
        return None
    
    # Crop the largest detected area
    x1, y1, x2, y2 = largest_bbox
    cropped_image = image[y1:y2, x1:x2]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    
    return output_path


if __name__ == "__main__":
    source = "./images/S__12861456_0.jpg"
    model_path = "./model/best_segment.pt"  
    output_cropped_path = "cropped_gauge.jpg"
    crop_largest_image = crop_largest_segment(source, model_path, output_cropped_path)
    print(f"save in path {crop_largest_image}")