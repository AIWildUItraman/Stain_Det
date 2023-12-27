import cv2
import numpy as np
import os

def stain_det(path:str):
    img = cv2.imread(path)
    img = img[0:1704, 209:2304]
    image_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([16, 150, 100])
    upper_yellow = np.array([26, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(img, img, mask=mask)
    transparent_red = np.array([0, 0, 200, 100], dtype=np.uint8)
    image_with_alpha[:,:,3] = 0

    image_with_alpha[mask == 0] = transparent_red

    return img, result
if __name__ == '__main__':

    # img_paths = os.listdir('examples')
    # print(img_paths)
 
    img_path = 'examples/WIN_20230822_14_53_05_Pro.jpg'
    img, result = stain_det(img_path)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary_result = cv2.threshold(gray_result, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    concatenated_image = np.concatenate((img, result), axis=1)
    # cv2.imshow('Resizable Window', img)
    cv2.imshow('Resizable Window', concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    