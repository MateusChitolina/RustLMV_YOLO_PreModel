import tkinter as tk
from tkinter import filedialog
from roboflow import Roboflow
import os

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    root.destroy()
    return file_path

def model_predict(pathModel):
    rf = Roboflow(api_key="13eIVpGfwizL7YIsh3cH")
    project = rf.workspace().project("corrosion-yolov8")
    model = project.version(4).model

    result = model.predict(
        pathModel,
        confidence=20,
        overlap=50
    )
    return result

def get_next_filename(base_name, ext):
    project_path = os.getcwd()
    i = 2
    while os.path.exists(os.path.join(project_path, f"{base_name}_{i}.{ext}")):
        i += 1
    return os.path.join(project_path, f"{base_name}_{i}.{ext}")

def get_extension(filename):
    return filename.split('.')[-1]