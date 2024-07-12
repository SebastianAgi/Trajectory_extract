import numpy as np
import os

#load data from a text file
def extract_data(file_path):
    data = []
    line_number = 0
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if parts:
                for part in parts:
                    try:
                        data.append(float(part))
                    except ValueError:
                        continue  # Skip the element if conversion fails
    return data


#extract data from the text file
file_path = '/home/sebastian/Documents/ANYmal_data/mine_hanheld_forest/OrbSLAM_data/times.txt'

data = extract_data(file_path)

start = data[0]

for i in range(0, len(data)):
    data[i] =  data[i] - start
    #convert to form like 6.342512e+02
    data[i] = "{:.6e}".format(data[i])

#write to a new file
new_file_path = '/home/sebastian/Documents/ANYmal_data/mine_hanheld_forest/OrbSLAM_data/correct_times_new.txt'
with open(new_file_path, 'w') as file:
    for line in data:
        file.write(line + '\n')

print(data[0:5])
print(len(data))