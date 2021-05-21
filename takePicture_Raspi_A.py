from picamera import PiCamera
import time
from datetime import datetime
from datetime import date


now = datetime.now()
t = now.strftime("%H%M%S")
today = date.today()
d = today.strftime("%y%m%d")

camera = PiCamera()
time.sleep(2)
camera.capture("/home/pi/Pictures/muestra" + d + t + ".jpg")
print("Done.")
