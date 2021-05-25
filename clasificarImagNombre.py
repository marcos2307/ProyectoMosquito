import glob
import os
newDir = 'Eliminados'

archivos = glob.glob('*.jpg')
for file_name in archivos:
    if int(file_name[13:17])>1600 or int(file_name[13:17])<800:
        os.rename(file_name, newDir + '/' + file_name)
        
for k in range(10):
    print(archivos[k][13:17])
