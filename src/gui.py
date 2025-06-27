from tkinter import Tk, Button, messagebox, Label, Toplevel, Canvas, Frame, Scale, HORIZONTAL
from utils import select_image, model_predict, start_webcam_analysis, stop_webcam

def run_prediction():
    fileNameModel = select_image()
    if not fileNameModel:
        messagebox.showinfo("Aviso", "Nenhuma imagem selecionada.")
        return

    result = model_predict(fileNameModel)
    
    if result:
        show_result_window(result)
    else:
        messagebox.showinfo("Info", "Não identificado.")

def create_gui():
    root = Tk()
    root.title("Detecção de ferrugens")
    root.geometry("400x300")

    title_label = Label(root, text="Sistema de Detecção de Ferrugem", 
                       font=("Arial", 16, "bold"))
    title_label.pack(pady=20)

    btn_predict = Button(root, text="Selecionar Imagem e Analisar", 
                        command=run_prediction, width=25, height=2)
    btn_predict.pack(pady=10)

    btn_webcam = Button(root, text="Análise ao Vivo (Webcam)", 
                       command=run_webcam_analysis, width=25, height=2,
                       bg='lightblue')
    btn_webcam.pack(pady=10)

    instructions = Label(root, text="Escolha uma opção:\n• Análise de imagem: Selecione um arquivo\n• Análise ao vivo: Use sua webcam",
                        justify='left', wraplength=350)
    instructions.pack(pady=20)

    return root

if __name__ == "__main__":
    root = create_gui()
    root.mainloop()

def show_result_window(result):
    result.plot()

webcam_active = False
current_cap = None

def run_webcam_analysis():
    global webcam_active, current_cap
    
    if webcam_active:
        messagebox.showinfo("Aviso", "Análise da webcam já está ativa.")
        return
    
    webcam_window = Toplevel()
    webcam_window.title("Análise ao Vivo - Detecção de Ferrugem")
    webcam_window.geometry("800x650")
    
    control_frame = Frame(webcam_window)
    control_frame.pack(pady=10)

    perf_frame = Frame(webcam_window)
    perf_frame.pack(pady=5)
    
    canvas = Canvas(webcam_window, width=640, height=480, bg='black')
    canvas.pack(pady=10)
    
    def start_analysis():
        global webcam_active, current_cap
        success, cap = start_webcam_analysis(canvas, skip_frames=2)
        if success:
            webcam_active = True
            current_cap = cap
            start_btn.config(state='disabled')
            stop_btn.config(state='normal')
            status_label.config(text="Status: Analisando ao vivo...")
        else:
            messagebox.showerror("Erro", f"Erro ao iniciar webcam: {cap}")
    
    def stop_analysis():
        global webcam_active, current_cap
        if current_cap:
            stop_webcam(current_cap)
            webcam_active = False
            current_cap = None
            start_btn.config(state='normal')
            stop_btn.config(state='disabled')
            status_label.config(text="Status: Parado")
            canvas.delete("all")
    
    def on_window_close():
        stop_analysis()
        webcam_window.destroy()
    
    start_btn = Button(control_frame, text="Iniciar Análise", command=start_analysis)
    start_btn.pack(side='left', padx=5)
    
    stop_btn = Button(control_frame, text="Parar Análise", command=stop_analysis, state='disabled')
    stop_btn.pack(side='left', padx=5)
    
    status_label = Label(webcam_window, text="Status: Pronto para iniciar")
    status_label.pack(pady=5)
    
    webcam_window.protocol("WM_DELETE_WINDOW", on_window_close)
