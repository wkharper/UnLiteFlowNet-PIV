from os import listdir, rename
from os.path import isfile, join

data_path = "./sample_data/cylinder_hd/"

dataset = [f for f in listdir(data_path) if isfile(join(data_path,f))]
dataset.sort()
print(dataset)
pairs = []
for i in range(0,len(dataset), 2):
    pairs.append((dataset[i-1], dataset[i]))

for i in range(0,len(pairs)):
    new_name_first = pairs[i][0].replace(
        pairs[i][0].split('_')[-1],
        "{:05d}".format(i) + "_img1.tif" 
    )
    new_name_second = pairs[i][1].replace(
        pairs[i][1].split('_')[-1],
        "{:05d}".format(i) + "_img2.tif" 
    )

    rename(data_path + pairs[i][0],data_path + new_name_first)
    rename(data_path + pairs[i][1],data_path + new_name_second)
    #pairs[i][0] = # rename to 05%i_img1
    #pairs[i][1] = #rename to 05%i_img2

#print(pairs)