import paho.mqtt.client as mqtt
from time import gmtime, strftime
def on_connect(client, userdata, flags, rc):
    print("Connect with result code " + str(rc))
    client.subscribe('ESP/ESP_A')
    client.subscribe('ESP/ESP_B')
    client.subscribe('Raspi/Raspi_A')
    client.subscribe('Raspi/Raspi_B')

def on_message(client, userdata, msg):
    t = strftime('%d%b%H%M', gmtime())
    f = open('C://muestras/' + msg.topic + '/muestra'+t+'.jpg', 'wb')
    f.write(msg.payload)
    f.close()

broker = '192.168.2.19'
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker)
client.loop_forever()
