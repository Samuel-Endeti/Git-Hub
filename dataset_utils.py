import os

def save_uploaded_images(files, class_name):
    folder = f"DATASET/{class_name}"
    os.makedirs(folder, exist_ok=True)

    for file in files:
        with open(os.path.join(folder, file.name), "wb") as f:
            f.write(file.getbuffer())

    return True