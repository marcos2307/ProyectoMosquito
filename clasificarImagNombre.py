import glob
import os
newDir = 'Eliminados'
if not(os.path.exists(newDir)):
    os.mkdir(newDir)

archivos = glob.glob('*.jpg')
for file_name in archivos:
    if int(file_name[13:17])>1630 or int(file_name[13:17])<800:
        os.rename(file_name, newDir + '/' + file_name)
        
for k in range(10):
    print(archivos[k][13:17])
