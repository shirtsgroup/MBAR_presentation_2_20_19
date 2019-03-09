from zipfile import ZipFile
import os
from shutil import rmtree

compress = True
top_dir = os.getcwd()
zip_file = str(os.getcwd()+'/data.zip')

if compress == True:
 if os.path.exists(zip_file): os.remove(zip_file)
 with ZipFile(zip_file,'w') as zip:
  for root, dirs, files in os.walk("./output"):
   for file in files:
    print(os.path.join(root,file))
    zip.write(os.path.join(root,file))
 zip.close()

if os.path.exists("./output"): 
 print("Removing folder 'output'")
 rmtree("./output")

with ZipFile(zip_file,'r') as zip:
 zip.extractall()
zip.close()  
