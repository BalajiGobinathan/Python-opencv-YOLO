
# Requirements: python3
# Packages: numpy opencv-python pyqt5

#importing all the necessary packages
import sys
import os
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QApplication,
                             QFileDialog, QLabel, QHBoxLayout)

from PyQt5.QtCore import (Qt)

import cv2
import numpy as np


def convert_video_to_frames(input_file_path, output_dir_path):
    cap = cv2.VideoCapture(input_file_path)

    if not cap.isOpened():
        return False, "Video Path is incorrect"

    file_paths = []

    idx = 0
    while True:
        
#reads the file and creates single frames and resizes it to 256*256 resolution and saves it in a folder
        
        ret, frame = cap.read()
        if ret is False:
            break

        resized_image = cv2.resize(frame, (256, 256))
        file_name = "%06d.png" % idx
        output_path = os.path.join(output_dir_path, file_name)
        file_paths.append(output_path)

        cv2.imwrite(output_path, resized_image)
        print("saving image at ", output_path)
        idx += 1

    return True, "Saved %d frames with success" % idx, file_paths

#gets the last output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def run_yolo(input_frames, output_folder):
#reads the class names from coco.names file and the configuration path also
    scale = 0.00392
    classes = None
    with open("coco.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    cfg_path, weigths_path = "yolov3.cfg", "yolov3.weights"
    net = cv2.dnn.readNet(weigths_path, cfg_path)
    idx = 0

    for file_path in input_frames:
        image = cv2.imread(file_path)
        
        #width and height
        Width = image.shape[1]
        Height = image.shape[0]

        #blob creation
        blob = cv2.dnn.blobFromImage(image, scale, (608,608), (0,0,0), True, crop=False)
        net.setInput(blob)

        #forward prpogation
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.3

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        #NMS - Non Maximum Suppression to avoid multiple bounding boxes for a single object
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        if len(boxes) > 0:
            for i in indices:
                i = i[0]
                box = boxes[i]
                class_id = class_ids[i]
                x = round(box[0])
                y = round(box[1])
                w = round(box[2])
                h = round(box[3])
                
                #drawing bounding boxes
                label = str(classes[class_id]) + " %.3f" % confidences[i]
                color = COLORS[class_id]
                cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
                cv2.putText(image, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        #the output after detection of images with bounding boxes are stored in the selected folder
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_folder, file_name)
        print("Saving detection at: ", output_file_path)
        cv2.imwrite(output_file_path, image)
        
        cv2.imshow("detect", image)
        cv2.waitKey(1)

        idx += 1
    
    return "Messages"
    

#for window, label, buttons and widget creation
class Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.resize(400, 400)
        self.setWindowTitle("DNN - Object Classifier")
        
        vbox = QVBoxLayout(self)
        vbox.setAlignment(Qt.AlignTop)


        self.txt_input_file = QLabel("")
        btn_select_video = QPushButton("Select Video")
        btn_select_video.clicked.connect(self.select_video_file)

        vbox.addWidget(QLabel("Input Video File:"))
        h1 = QHBoxLayout()
        h1.addWidget(self.txt_input_file)
        h1.addWidget(btn_select_video), btn_select_video.adjustSize()
        vbox.addLayout(h1)


        self.txt_single_path = QLabel("")
        btn_select_frames = QPushButton("Select Folder")
        btn_select_frames.clicked.connect(self.select_output_folder)

        vbox.addWidget(QLabel("Output Single Raw Frames Path:"))
        h2 = QHBoxLayout()
        h2.addWidget(self.txt_single_path)
        h2.addWidget(btn_select_frames), btn_select_frames.adjustSize()
        vbox.addLayout(h2)
        

        self.txt_output_folder_path = QLabel("")
        btn_select_detections = QPushButton("Select Folder")
        btn_select_detections.clicked.connect(self.select_detections_folder)

        vbox.addWidget(QLabel("Output Detections Path:"))
        h3 = QHBoxLayout()
        h3.addWidget(self.txt_output_folder_path)
        h3.addWidget(btn_select_detections), btn_select_detections.adjustSize()
        vbox.addLayout(h3)

        btn_run = QPushButton("Run")
        btn_run.clicked.connect(self.run_detections)
        vbox.addWidget(btn_run)
        
        self.txt_status = QLabel("")
        vbox.addWidget(self.txt_status)
        
        self.show()

    def select_video_file(self):
        file_path = QFileDialog.getOpenFileName(None, "Open Video File",
                                                    None,
                                                    filter='All Files (*.*)')
        if file_path:
            self.txt_input_file.setText(file_path[0])

    def select_output_folder(self):
        open_folder = QFileDialog.getExistingDirectory(None, "Open Output Single Frames Folder")
        if open_folder:
            self.txt_single_path.setText(open_folder)
    
    def select_detections_folder(self):
        open_folder = QFileDialog.getExistingDirectory(None, "Open Ouput Detections Folder")
        if open_folder:
            self.txt_output_folder_path.setText(open_folder)

    def run_detections(self):

        single_frames_path = self.txt_single_path.text()
        detections_path = self.txt_output_folder_path.text()
        video_path = self.txt_input_file.text()

        #Error messages if path, file, folder does not exist
        if not os.path.exists(video_path):
            self.txt_status.setText("Fail: input file does not exist!")
            return

        if not os.path.isdir(single_frames_path):
            self.txt_status.setText("Fail: output single frames folder does not exist!")
            return

        if not os.path.isdir(detections_path):
            self.txt_status.setText("Fail: output detections folder does not exist!")
            return

        ret, message, file_paths = convert_video_to_frames(video_path,
                                                           single_frames_path)
        if not ret:
            self.txt_status.setText(message)
            return

        
        ret, message = run_yolo(file_paths, detections_path)
        self.txt_status.setText(message)
        
            

        #closes all the video displaying, the widget and all other windows once detection is completed
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


