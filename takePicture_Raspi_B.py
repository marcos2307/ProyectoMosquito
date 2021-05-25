from picamera import PiCamera
import time
from datetime import datetime
from datetime import date
import paho.mqtt.client as mqtt

now = datetime.now()
t = now.strftime("%H%M%S")
today = date.today()
d = today.strftime("%y%m%d")

camera = PiCamera()
time.sleep(2)
camera.capture('/home/pi/Pictures/muestra' + d + t + '.jpg')

def on_connect(client, userdata, flags, rc):
    print('Connect with result code ' + str(rc))
def on_publish(mosq, userdata, mid):
    mosq.disconnect()
    
broker = '192.168.2.15'
client = mqtt.Client()
client.on_connect = on_connect
client.on_publish = on_publish
client.connect(broker)

f=open("/home/pi/Pictures/muestra" + d + t + ".jpg", "rb") #3.7kiB in same folder
fileContent = f.read()
byteArr = bytearray(fileContent)
client.publish("Raspi/Raspi_B",byteArr,0)

client.loop_forever()
