import pandas as pd
import os
import shutil

csv_path = "dataset/train.csv"
image_folder = "dataset/train/train"
output_folder = "dataset/organized"

df = pd.read_csv(csv_path)

for label in df["Label"].unique():
    os.makedirs(os.path.join(output_folder, label), exist_ok=True)

for _, row in df.iterrows():
    image_name = row["Id"]
    label = row["Label"]

    source = os.path.join(image_folder, image_name)
    destination = os.path.join(output_folder, label, image_name)

    if os.path.exists(source):
        shutil.copy2(source, destination)

print("Dataset successfully organized!")