import shutil
import os
import sys

source = '/home/hnko/Desktop/FAA_PROJECT/out/more/'
dest = '/home/hnko/Desktop/FAA_PROJECT/out/'

files = os.listdir('/home/hnko/Desktop/FAA_PROJECT/out/more')
# import pdb; pdb.set_trace()
i = 2000
for filename in files:
    new_name = f'{i}_{filename[0]+filename[-4:]}'
    shutil.move(source + filename, dest + new_name)
    i+=1
os.rmdir(source)
