import pandas as pd
import os
import numpy as np
from PIL import Image
from datetime import datetime

#TODO learn pickle and use it to save and load the dataframes
#!!! too slow. don't read png files for every input
#!!! cant store the png files in memory because of the size
#!!! try algorithmic approach
#!!! chunk 24 data into 700s and 24 data into 100s

def read_png_to_ndarray(png_path):
    # Open the image file
    with Image.open(png_path) as img:
        # Convert the image to a NumPy array
        img_array = np.array(img)
    return img_array

def compare_PNG(png1,png2):
    if png1.shape != png2.shape:
        return False
    elif (png1 == png2).all():
        return True
    else:
        return False

def to_PNG(dicom_path):
    dicom_path = os.path.splitext(dicom_path)[0] + ".png"
    return dicom_path



df_24 = pd.read_csv("../Teknofest2024_annotation.csv")
df_23 = pd.read_csv("../Teknofest2023_paths.csv")


data_cnt_24 = len(df_24["dicom_path"]) #5441 index
data_cnt_23 = len(df_23["dicom_path"]) #30722 index

chunk_size_24 = 700
chunk_size_23 = 10

matches = ["0"] * data_cnt_24
matched_files = ["0"] * data_cnt_24

chunk_counter_24 = 2 #* 24 data will be divided into 700s

while chunk_counter_24 * chunk_size_24 < data_cnt_24:

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Current time: {current_time}")
    print(f"Processing chunk {chunk_counter_24}")
    chunk_counter_23 = 0 #* 23 data will be divided into 100s

    #* read 24 data chunk
    files_24 = []
    for i in range((chunk_counter_24) * chunk_size_24, (chunk_counter_24 + 1) * chunk_size_24): #0 to 699 is the start
        if i >= data_cnt_24:
            break
        path = df_24["dicom_path"][i]
        path = os.path.splitext(path)[0] + ".png"
        try:
            png_24 = read_png_to_ndarray(path)
            files_24.append(png_24)
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            files_24.append(np.zeros((1,1)))

    while chunk_counter_23 * chunk_size_23 < data_cnt_23:

        current_time = datetime.now().strftime("%H:%M:%S")
        if chunk_counter_23 % 100 == 0:
            print(f"    Current time: {current_time}")
            print(f"    Processing chunk {chunk_counter_23}")

        #* read 23 data chunk
        files_23 = []
        for j in range((chunk_counter_23) * chunk_size_23, (chunk_counter_23 + 1) * chunk_size_23):
            if j >= data_cnt_23:
                break
            try:
                png_23 = read_png_to_ndarray(df_23["dicom_path"][j])
                files_23.append(png_23)
            except Exception as e:
                print(f"Error reading file {df_23['dicom_path'][j]}: {e}")
                files_23.append(np.ones((1,1)))
        
        #* compare two chunks and update matches
        for i, png_24 in enumerate(files_24):
            try:

                for j, png_23 in enumerate(files_23):
                    try:

                        if compare_PNG(png_24, png_23):
                            index_24 = chunk_counter_24 * chunk_size_24 + i
                            index_23 = chunk_counter_23 * chunk_size_23 + j
                            matches[index_24] = "1"
                            matched_files[index_24] = df_23["dicom_path"][index_23]
                            print(f"        Match found: {index_24} - {index_23}")
                            break

                    except Exception as e:
                        print(f"Error comparing files: {e} at chunk {chunk_counter_24} and {chunk_counter_23}")

            except Exception as e:
                print(f"Error comparing files: {e} at chunk {chunk_counter_24} and {chunk_counter_23}")

        #* go to next chunks
        chunk_counter_23 += 1
    chunk_counter_24 += 1

print("Creating CSV")

df_24["matches"] = matches
df_24["matched_files"] = matched_files
df_24.to_csv("../Teknofest2024_annotation.csv", index=False)

print("Done")