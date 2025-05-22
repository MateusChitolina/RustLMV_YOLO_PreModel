from tkinter import Tk, Button, messagebox, Label, Toplevel
from utils import select_image, model_predict
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def run_prediction():
    fileNameModel = select_image()
    if not fileNameModel:
        messagebox.showinfo("Aviso", "Nenhuma imagem selecionada.")
        return

    result = model_predict(fileNameModel)
    
    if result:
        show_result_window(result)
    else:
        messagebox.showinfo("Erro", "Falha ao realizar a previsão.")

def create_gui():
    root = Tk()
    root.title("Detecção de ferrugens")
    root.geometry("300x200")

    btn_predict = Button(root, text="Selecionar Imagem e Prever", command=run_prediction)
    btn_predict.pack(pady=20)

    return root

if __name__ == "__main__":
    root = create_gui()
    root.mainloop()

def show_result_window(result):
    result.plot()