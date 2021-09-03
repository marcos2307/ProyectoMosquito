import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(05,GPIO.OUT)

def recargar():
	GPIO.output(05, GPIO.HIGH)
	sleep(2)
	GPIO.output(05, GPIO.LOW)
	sleep(2)
recargar()
GPIO.cleanup()
