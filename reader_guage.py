import cv2
import numpy as np
import math

def process_gauge(image_path, min_angle=45, max_angle=315, min_value=0, max_value=100):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=50, maxRadius=150)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            # Detecting lines (needle)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
            if lines is not None:
                max_line = None
                max_length = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > max_length:
                        max_length = length
                        max_line = (x1, y1, x2, y2)
                
                if max_line:
                    x1, y1, x2, y2 = max_line
                    
                    # Calculate the angle of the needle
                    angle = math.degrees(math.atan2(y1 - center[1], x1 - center[0]))
                    
                    # Normalize the angle (Adjust for the gauge scale range)
                    angle = angle if angle >= 0 else 360 + angle
                    
                    # Adjusting for the gauge's zero position and scale
                    adjusted_angle = (angle - min_angle) % 360
                    
                    # Ensure the angle is within the valid gauge range
                    if adjusted_angle < 0:
                        adjusted_angle += 360
                    
                    if adjusted_angle > (max_angle - min_angle):
                        return None  # Out of valid range
                    
                    # Map angle to gauge reading
                    reading = ((adjusted_angle / (max_angle - min_angle)) * (max_value - min_value)) + min_value
                    return reading
    
    return None

# Example usage
image_path = 'cropped_gauge.jpg'
gauge_value = process_gauge(image_path, min_angle=0, max_angle=360, min_value=0, max_value=2500)
print(f'Gauge Reading: {gauge_value:.2f}')
