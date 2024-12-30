import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

def percent_to_u8(percent):
    return int((float(percent) / 100.0) * 255)

@dataclass
class TorusDetectorOptions:
    hsv_start: Tuple[int, int, int] = (0, percent_to_u8(10), percent_to_u8(60))
    hsv_end: Tuple[int, int, int] = (30, percent_to_u8(100), percent_to_u8(100))
    blur_kernel_size: int = 7

    # Add options for rectangle detection
    aspect_ratio_range: Tuple[float, float] = (4, 6)

class TorusDetector:
    def __init__(self, config: Optional[TorusDetectorOptions] = None):
        if config is not None:
            self.config = config
        else:
            self.config = TorusDetectorOptions()

    def generate_color_mask(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.hsv_start, self.config.hsv_end)
        return mask
    
    def remove_noise(self, mask):
        kernel = np.ones((5, 5), np.uint8) 
        erosion = cv2.erode(mask, kernel)
        erosion = cv2.erode(erosion, kernel)
        erosion = cv2.erode(erosion, kernel)
        dialation = cv2.dilate(erosion, kernel)
        return dialation


    def detect_rectangles(self, image_gray):
        _, thresh = cv2.threshold(image_gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            if self.config.aspect_ratio_range[0] <= aspect_ratio <= self.config.aspect_ratio_range[1]:
                rectangles.append((x, y, w, h))

        return rectangles

    def __call__(self, input_image):
        mask = self.generate_color_mask(input_image)
        finalmask = self.remove_noise(mask)
        result = cv2.bitwise_and(input_image, 2_image, mask=finalmask)
        result = cv2.medianBlur(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), self.config.blur_kernel_size)

        rectangles = self.detect_rectangles(result)
        return rectangles

if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    td = TorusDetector()

    while True:
        ret, image = vid.read()
        rectangles = td(image)

        # Visualize the rectangles.
        blob_color = (255, 0, 0)
        blobs = image.copy()
        count = 0

        if rectangles:
            count = len(rectangles)
            for rect in rectangles:
                x, y, w, h = rect
                cv2.rectangle(blobs, (x, y), (x + w, y + h), blob_color, 2)

        
        text = "Number of Detected Rectangles: " + str(count)
        cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, blob_color, 2)

        # Show image
        cv2.imshow("Filtering Rectangular Blobs Only", blobs)
        if cv2.waitKey(10) == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
