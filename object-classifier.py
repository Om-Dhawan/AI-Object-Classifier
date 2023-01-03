import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk

from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL


class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):
        img_list = np.array([])
        class_list = np.array([])

        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg')[:, :, 0]
            img = img.reshape(16950)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg')[:, :, 0]
            img = img.reshape(16950)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        img_list = img_list.reshape(counters[0] - 1 + counters[1] - 1, 16950)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg')[:, :, 0]
        img = img.reshape(16950)
        prediction = self.model.predict([img])

        return prediction[0]



class Camera:

    def __init__(self):
        self.camera = cv.VideoCapture(0)
        if not self.camera.isOpened():
            raise ValueError("Unable to open camera!")

        self.width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.camera.isOpened():
            for folder in ['1', '2']:
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            os.rmdir("1")
            os.rmdir("2")
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return None
        
        

class App:

    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):

        self.window = window
        self.window_title = window_title

        self.counters = [1, 1]

        self.model = Model()

        self.auto_predict = False

        self.camera = Camera()

        self.init_gui()

        self.delay = 15
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):

        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=lambda: self.model.train_model(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predcit", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="OUTPUT")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)


    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        cv.imwrite(f'{class_num}/frame{self.counters[class_num-1]}.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')

        self.counters[class_num - 1] += 1

    def reset(self):
        for folder in ['1', '2']:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        self.counters = [1, 1]
        self.model = Model()
        self.class_label.config(text="OUTPUT")

    def update(self):
        if self.auto_predict:
            print(self.predict())

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)

        if prediction == 1:
            self.class_label.config(text=self.classname_one)
            return self.classname_one
        if prediction == 2:
            self.class_label.config(text=self.classname_two)
            return self.classname_two
        
        
        
        
def main():
    App(window_title="Camera Classifier")

if __name__ == "__main__":
    main()