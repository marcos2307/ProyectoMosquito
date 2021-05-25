import glob
import os

archivos = glob.glob('/home/pi/Pictures/*.jpg')
for file_name in archivos:
    os.remove(file_name)
print("Todas las imágenes de /home/pi/Pictures/ han sido eliminadas.")