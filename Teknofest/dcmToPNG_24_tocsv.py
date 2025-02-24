import os
import numpy as np
import pandas as pd
from TT_dicom2png import DICOMConverter
import pydicom

inputdirroot = '../Teknofest-2024/'
#outdir = '../Teknofest_PNG/Kategori5Sol_png/'
outdirroot = '../Teknofest_2024/'

dirs = ["Kategori1", "Kategori1_2", "Kategori2Sag", "Kategori2Sol", "Kategori4Sag", "Kategori4Sol", "Kategori5Sag", "Kategori5Sol"]

csvdata = []


def birads(basepath):
    category = os.path.basename(basepath)
    if category == "Kategori1" or category == "Kategori1_2":
        return "BIRADS_1"
    elif category == "Kategori2Sag" or category == "Kategori2Sol":
        return "BIRADS_2"
    elif category == "Kategori4Sag" or category == "Kategori4Sol":
        return "BIRADS_4"
    elif category == "Kategori5Sag" or category == "Kategori5Sol":
        return "BIRADS_5"
    

def process_dicom_files(id_dirs, outdir, inputdir):
    converter = DICOMConverter(inputdir)
    file_count = 0
    for id_dir in id_dirs:
        file_count += 1
        print(f"Processing directory {file_count}, %{file_count/len(id_dirs)*100} is finished: {id_dir}")
        input_path = os.path.join(inputdir, id_dir)
        dicom_files = [f for f in os.listdir(input_path) if f.endswith('.dcm')]

        os.makedirs(outdir, exist_ok=True)

        for dicom_file in dicom_files:
            try:
                dicom_relative_path = os.path.join(id_dir, dicom_file)
                
                converter.convert_to_png(dicom_relative_path, outdir)

                dicom_path = os.path.join(inputdir, dicom_relative_path)
                img = pydicom.read_file(dicom_path)
                study_instance_uid = img.StudyInstanceUID
                series_instance_uid = img.SeriesInstanceUID
                sop_instance_uid = img.SOPInstanceUID
                patient_id = img.PatientID
                category = birads(inputdir)
                rows = img.Rows
                columns = img.Columns

                csvdata.append([dicom_path, study_instance_uid, series_instance_uid, sop_instance_uid, patient_id, category, rows, columns])

            except Exception as e:
                print(f"Error processing file {dicom_file} in directory {id_dir}: {e}")

for href in dirs:
    inputdir = os.path.join(inputdirroot, href)
    outdir = os.path.join(outdirroot, href)

    os.makedirs(outdir, exist_ok=True)

    id_dirs = []
    entries = os.listdir(inputdir)
    # Iterate over all entries
    for entry in entries:
        full_path = os.path.join(inputdir, entry)
        # If entry is a directory, add it to the list
        if os.path.isdir(full_path):
            id_dirs.append(entry)
    print(f"Training directories: {len(id_dirs)}")

    print(f"Processing dicom directories of {href} ...")
    process_dicom_files(id_dirs, outdir, inputdir)

df = pd.DataFrame(csvdata, columns=["dicom_path", "study_instance_uid", "series_instance_uid", "sop_instance_uid", "patient_id", "category", "rows", "columns"])

df.to_csv("../Teknofest2024_annotation.csv", index=True)

print("Done!")

