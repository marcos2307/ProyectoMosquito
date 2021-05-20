import paho.mqtt.client as mqtt
from time import gmtime, strftime
def on_connect(client, userdata, flags, rc):
    print("Connect with result code " + str(rc))
    client.subscribe('PICTURE_A')

def on_message(client, userdata, msg):
    t = strftime('%d%b%H%M', gmtime())
    f = open('muestra'+t+'.jpg', 'wb')
    f.write(msg.payload)
    f.close()

broker = '192.168.2.15'
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker)
client.loop_forever()
