import pandas as pd
import re
#* functions
def take_id(path):
    parts = path.split('/')
    path = parts[-2]
    return int(path)

def take_id_path(path):
    parts = path.split('/')
    path = '/'.join(parts[-2:])
    if path.endswith('.dcm'):
        path = path.replace('.dcm', '.png')
    return path

def take_category_xlsx(strng):
    match = re.search(r'Kategori (\d+):', strng)
    if match:
        return int(match.group(1))
    return None

#* variable initialization
df_23_old = pd.read_csv("../veribilgisi23.csv")
df_23_paths = pd.read_csv("../Teknofest2023_paths.csv")
df_24 = pd.read_csv("../Teknofest2024_annotation.csv")
mdai_full = pd.read_csv("../mdai_full.csv")
veribilgisi_xlsx = pd.read_excel("../labellar_full.xlsx")

year = []
path = []
patient_id = []
study_instance_id = []
series_id = []
sop_id = []
birads_category = []
exist_in_2023 = []
exist_in_2024 = []
matched_index = []
matched_path = []

#* preparation
id_dict = {}
for i in range(len(mdai_full)):
    id_dict[take_id(mdai_full["input_path"][i])] = [
        mdai_full["StudyInstanceUID"][i],
        mdai_full["SeriesInstanceUID"][i],
        mdai_full["SOPInstanceUID"][i],
        #int(mdai_full["BiradsScore"][i][-1])
    ]

birads_dict = {}
for i in range(len(veribilgisi_xlsx)):
    birads_dict[veribilgisi_xlsx["patient_id"][i]] = [
        take_category_xlsx(veribilgisi_xlsx["birads_score"][i]),
    ]

matched_dict = {}
for i in range(len(df_24)):
    if int(df_24["matches"][i]) == 1:
        matched_dict[df_24["dicom_path"][i]] = df_24["matched_files"][i]
        matched_dict[df_24["matched_files"][i]] = df_24["dicom_path"][i]

"""#* checking
print(len(mdai_full))
i = 0
for path in df_23_paths["dicom_path"]:
    p_id = take_id(path)
    if p_id not in id_dict:
        #print(p_id)
        i += 1
print(i)
print("xxxx")
quit()"""

#* creating csv
len23 = len(df_23_paths)
len24 = len(df_24)

print("23 data processing")
index_dict = {}
passed_birads0 = 0
for i in range(len23):
    png_path = df_23_paths["dicom_path"][i]
    p_id = take_id(png_path)
    birads_class = birads_dict[p_id][0]
    if birads_class == 0:
        passed_birads0 += 1
        continue

    year.append(23)
    path.append(png_path)
    patient_id.append(p_id)
    if p_id in id_dict:
        study_instance_id.append(id_dict[p_id][0])
        series_id.append(id_dict[p_id][1])
        sop_id.append(id_dict[p_id][2])
    else:
        study_instance_id.append("")
        series_id.append("")
        sop_id.append("")
    birads_category.append(birads_dict[p_id][0])

    exist_in_2023.append(0)
    if png_path in matched_dict:
        exist_in_2024.append(1)
        matched_path.append(matched_dict[png_path])
    else:
        exist_in_2024.append(0)
        matched_path.append("")

    matched_index.append(-1)
    index_dict[png_path] = i - passed_birads0
print(len23)
print(len(year), len(path), len(patient_id), len(study_instance_id), len(series_id), len(sop_id), len(birads_category), len(exist_in_2023), len(exist_in_2024), len(matched_index), len(matched_path))

print("24 data processing")
for i in range(len24):
    index24 = len23 + i
    year.append(24)
    png_path = df_24["dicom_path"][i]
    path.append(png_path)

    p_id = take_id(png_path)
    patient_id.append(p_id)
    study_instance_id.append(df_24["study_instance_uid"][i])
    series_id.append(df_24["series_instance_uid"][i])
    sop_id.append(df_24["sop_instance_uid"][i])
    birads_category.append(int(df_24["category"][i][-1]))

    if df_24["matches"][i] == 1:
        exist_in_2023.append(1)
        matched_path.append(df_24["matched_files"][i])
        matched_index.append(index_dict[df_24["matched_files"][i]])
        matched_index[index_dict[df_24["matched_files"][i]]] = index24
    else:
        exist_in_2023.append(0)
        matched_path.append("")
        matched_index.append(-1)
    exist_in_2024.append(0)

print(len(year), len(path), len(patient_id), len(study_instance_id), len(series_id), len(sop_id), len(birads_category), len(exist_in_2023), len(exist_in_2024), len(matched_index), len(matched_path))

myset = set()
myset.add(len(year))
myset.add(len(path))
myset.add(len(patient_id))
myset.add(len(study_instance_id))
myset.add(len(series_id))
myset.add(len(sop_id))
myset.add(len(birads_category))
myset.add(len(exist_in_2023))
myset.add(len(exist_in_2024))
myset.add(len(matched_index))
myset.add(len(matched_path))

if len(myset) != 1:
    print("error")
    quit()

df_final = pd.DataFrame({
    "year": year,
    "path": path,
    "patient_id": patient_id,
    "study_instance_uid": study_instance_id,
    "series_instance_uid": series_id,
    "sop_instance_uid": sop_id,
    "birads_category": birads_category,
    "match_exist_in_2023": exist_in_2023,
    "match_exist_in_2024": exist_in_2024,
    "matched_index": matched_index,
    "matched_path": matched_path
})

df_final.to_csv("../Teknofest_final.csv", index=True)
print("done")