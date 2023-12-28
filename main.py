import cv2
import numpy as np
import scripts.utils as utils




def main():
    # read from camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
        img, mask, stain_ratio = utils.stain_det(frame)
        

        if mask[100, 100] and mask[100, 150]:
            stain_ratio_text = f"Stain Ratio: {stain_ratio:.2%}"
            cv2.putText(frame, stain_ratio_text, (100, 100), 
                        cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0), 4)
        
        

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




