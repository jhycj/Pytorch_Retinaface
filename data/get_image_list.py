import os 
import glob

data_folder = './data/FDDB/images/train/0/*'
output_file = './data/FDDB/img_list.txt' 

file_list = glob.glob(data_folder)

with open (output_file, 'w') as f: 
    
    for file_path in file_list: 
        file_name = file_path.split('/')[-1].split('.jpg')[0]
        f.write(file_name) 
        f.write('\n') 
f.close()  