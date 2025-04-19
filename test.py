import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os

class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Face Detector")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Initialize variables
        self.model = YOLO("best.pt")
        self.current_image = None
        self.photo = None

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create image display area
        self.canvas = tk.Canvas(self.main_frame, bg="white", width=640, height=480)
        self.canvas.pack(pady=10)

        # Create button frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        # Style for buttons
        style = ttk.Style()
        style.configure("Action.TButton", padding=10)

        # Create buttons
        self.load_btn = ttk.Button(
            self.button_frame,
            text="Load Image",
            command=self.load_image,
            style="Action.TButton"
        )
        self.predict_btn = ttk.Button(
            self.button_frame,
            text="Predict",
            command=self.predict,
            style="Action.TButton"
        )
        self.reset_btn = ttk.Button(
            self.button_frame,
            text="Reset",
            command=self.reset,
            style="Action.TButton"
        )

        # Pack buttons
        self.load_btn.pack(side=tk.LEFT, padx=5)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)

    def display_image(self, image):
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = 640
            canvas_height = 480
            image_width, image_height = image_pil.size
            ratio = min(canvas_width/image_width, canvas_height/image_height)
            new_size = (int(image_width*ratio), int(image_height*ratio))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image_pil)
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width//2,
                canvas_height//2,
                image=self.photo,
                anchor="center"
            )

    def predict(self):
        if self.current_image is not None:
            results = self.model.predict(self.current_image, conf=0.25)
            predicted_image = results[0].plot()
            self.display_image(predicted_image)

    def reset(self):
        if self.current_image is not None:
            self.display_image(self.current_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()
