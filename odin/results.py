
import cv2


class YOLOResult:
    def __init__(self, image_with_bboxes, inference_speed):
        self.image_with_bboxes = image_with_bboxes
        self.inference_speed = inference_speed
        
    def show(self):
        cv2.imshow("Inference Result", self.image_with_bboxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()