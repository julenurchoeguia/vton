import os 

path = "/var/hub/VITON-HD-results-ladi-vton/paired/upper_body"
text_files_path = "/var/hub/VITON-HD-results-ladi-vton/"

list_of_files = os.listdir(path)

train_files , test_files, val_files = [], [], []

for i, file in enumerate(list_of_files):
    if i % 10 == 0:
        test_files.append(file)
    elif i % 10 == 1:
        val_files.append(file)
    else:
        train_files.append(file)


print(len(train_files), len(test_files), len(val_files))

with open(text_files_path + "train_files.txt", "w") as f:
    for file in train_files:
        f.write(file + "\n")

with open(text_files_path + "test_files.txt", "w") as f:
    for file in test_files:
        f.write(file + "\n")

with open(text_files_path + "val_files.txt", "w") as f:
    for file in val_files:
        f.write(file + "\n")
