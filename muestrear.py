import cv2
import numpy as np

from datetime import datetime
from datetime import date

def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    display_width=3264,
    display_height=2464,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #avg_a = np.average(result[:, :, 1])
    #avg_b = np.average(result[:, :, 2])
    #result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    #result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    #result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def red_balance(img):
    result = img
    avg_b = np.average(result[:, :, 0])
    avg_g = np.average(result[:, :, 1])
    avg_r = np.average(result[:, :, 2])
    b = 128
    r = .5  
    result[:, :, 0] = (result[:, :, 0]/avg_b)*b
    result[:, :, 1] = (result[:, :, 1]/avg_g)*b
    result[:, :, 2] = (result[:, :, 2]/avg_r)*b
    return result

def take_picture(name):
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        cv2.waitKey(10)
        ret_val, img = cap.read()
        result = img
        #cv2.namedWindow("resultado", cv2.WINDOW_NORMAL) 
        #cv2.imshow("resultado", result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite("./muestras/" + name+'.png', result)
        cap.release()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    now = datetime.now()
    t = now.strftime("%H%M%S")
    today = date.today()
    d = today.strftime("%y%m%d")
    take_picture("muestra" + d + t)
    now = datetime.now()
    t = now.strftime("%H%M%S")
    take_picture("muestra" + d + t)

