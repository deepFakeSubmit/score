import sys
import os

g = os.walk("../zju_deepfake/eval2/flac/")  
f = open("../zju_deepfake/eval2.txt", 'w')

for path,dir_list,file_list in g:  
    for file_name in file_list:
        file_name =  file_name.replace(".flac", "")
        f.write("%s spoof\n"%file_name)  
        print(file_name)