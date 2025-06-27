import tkinter as tk
from tkinter import filedialog
from roboflow import Roboflow
import os
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time

API_KEY = "YOUR_ROBOFLOW_API_KEY"
class AsyncVideoCapture:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = False
        self.thread = None
        self.last_frame = None
        self.frame_count = 0
        
    def start(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.src, backend)
                if self.cap.isOpened():
                    print(f"AsyncVideoCapture: Inicializado com backend {backend}")
                    break
                else:
                    if self.cap:
                        self.cap.release()
            except Exception as e:
                print(f"Erro ao tentar backend {backend}: {e}")
                continue
        
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Não foi possível inicializar a câmera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        except:
            pass
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print(f"AsyncVideoCapture: Configurado para {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self.cap.get(cv2.CAP_PROP_FPS)} FPS")
        
    def _capture_loop(self):
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    consecutive_failures = 0
                    self.frame_count += 1
                    self.last_frame = frame.copy()
                    
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass 
                        
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("AsyncVideoCapture: Muitas falhas consecutivas, tentando reiniciar...")
                        self._restart_capture()
                        consecutive_failures = 0
                    
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"AsyncVideoCapture: Erro na captura: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.05)
                
    def _restart_capture(self):
        try:
            if self.cap:
                self.cap.release()
            time.sleep(0.1)
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                print("AsyncVideoCapture: Captura reiniciada com sucesso")
        except Exception as e:
            print(f"AsyncVideoCapture: Erro ao reiniciar: {e}")
    
    def read(self):
        if not self.running:
            return False, None
            
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None
    
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened() and self.running
    
    def get_fps_info(self):
        return {
            'frames_captured': self.frame_count,
            'queue_size': self.frame_queue.qsize(),
            'is_running': self.running
        }
    
    def stop(self):
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Limpar queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.cap:
            self.cap.release()
            
        print("AsyncVideoCapture: Parado e recursos liberados")

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    root.destroy()
    return file_path

def model_predict(pathModel):
    rf = Roboflow(api_key=API_KEY)
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

def start_webcam_analysis(canvas, confidence_threshold=20, overlap_threshold=50, skip_frames=3):
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project("corrosion-yolov8")
    model = project.version(4).model
    
    try:
        async_cap = AsyncVideoCapture(0)
        async_cap.start()
    except Exception as e:
        return False, f"Erro ao inicializar captura assíncrona: {e}"
    
    frame_count = 0
    last_predictions = []
    processing = False
    last_process_time = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    def predict_async(frame):
        nonlocal last_predictions, processing, last_process_time
        
        if processing or (time.time() - last_process_time) < 1.5:
            return
        
        processing = True
        last_process_time = time.time()
        
        try:
            small_frame = cv2.resize(frame, (320, 240))
            
            temp_path = "temp_frame_small.jpg"
            cv2.imwrite(temp_path, small_frame)
            
            result = model.predict(
                temp_path,
                confidence=confidence_threshold,
                overlap=overlap_threshold
            )
            
            scale_x = 640 / 320
            scale_y = 480 / 240
            
            new_predictions = []
            if hasattr(result, 'predictions') and result.predictions:
                for prediction in result.predictions:
                    try:
                        pred_data = prediction.json()

                        x = int((pred_data['x'] - pred_data['width']/2) * scale_x)
                        y = int((pred_data['y'] - pred_data['height']/2) * scale_y)
                        w = int(pred_data['width'] * scale_x)
                        h = int(pred_data['height'] * scale_y)
                        
                        new_predictions.append({
                            'x': max(0, x),
                            'y': max(0, y),
                            'w': min(w, 640-x),
                            'h': min(h, 480-y),
                            'class': pred_data['class'],
                            'confidence': pred_data['confidence']
                        })
                    except Exception as pred_error:
                        print(f"Erro ao processar predição: {pred_error}")
                        continue
            
            last_predictions = new_predictions
            if new_predictions:
                print(f"AsyncCapture: {len(new_predictions)} detecções processadas")
            

            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Erro na predição: {e}")
        finally:
            processing = False
    
    def update_frame():
        nonlocal frame_count, fps_counter, fps_start_time
        
        if not async_cap.is_opened():
            print("AsyncCapture: Captura não está ativa")
            canvas.after(100, update_frame)
            return
        
        ret, frame = async_cap.read()
        
        if not ret or frame is None:
            canvas.after(16, update_frame)
            return
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_count % (skip_frames * 15) == 0 and not processing:
                thread = threading.Thread(target=predict_async, args=(frame.copy(),), daemon=True)
                thread.start()
            
            for pred in last_predictions:
                try:
                    cv2.rectangle(frame_rgb, (pred['x'], pred['y']), 
                                 (pred['x'] + pred['w'], pred['y'] + pred['h']), (0, 255, 0), 2)
                    
                    label = f"{pred['class']}: {pred['confidence']:.2f}"
                    cv2.putText(frame_rgb, label, (pred['x'], pred['y']-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception:
                    continue
            
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:  # A cada segundo
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
                
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame_rgb, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                fps_info = async_cap.get_fps_info()
                info_text = f"Queue: {fps_info['queue_size']}, Frames: {fps_info['frames_captured']}"
                cv2.putText(frame_rgb, info_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 480))
            photo = ImageTk.PhotoImage(frame_pil)
            
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo
            
            frame_count += 1
            
        except Exception as display_error:
            print(f"Erro ao exibir frame: {display_error}")
        
        canvas.after(16, update_frame)
    
    update_frame()
    return True, async_cap

def stop_webcam(async_cap):
    try:
        if async_cap and hasattr(async_cap, 'stop'):
            async_cap.stop()
        cv2.destroyAllWindows()
        print("Captura assíncrona parada com sucesso")
    except Exception as e:
        print(f"Erro ao parar captura assíncrona: {e}")
