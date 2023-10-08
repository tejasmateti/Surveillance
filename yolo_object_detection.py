import cv2
import numpy as np
import glob
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email(label):
    fromaddr = "tejasmateti1@gmail.com"  # sender gmail address
    toaddr = "tejasmateti22@gmail.com"  # reciver gmail address
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Unauthorized entry in restricted area"

    body = ""
    filename = ""

    if(str(label)=="human"):
      body = "Human entered restricted area. \nBelow is the image captured from camera 1"
      filename ="Human.jpg"
    else:
      body = "A leopard entered restricted area. \nBelow is the image captured from camera 1"
      filename = "Leopard.jpg"

    msg.attach(MIMEText(body, 'plain'))

    attachment = open(r"C:\Users\suraj\OneDrive\Desktop\Chetah images\301.jpg", "rb")  # image folder

    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())

    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename=%s" % filename)
    msg.attach(p)

    s = smtplib.SMTP('smtp.gmail.com', 587)

    s.starttls()

    s.login(fromaddr, "mefuwglvepxhhlyi")  # enter sender gmail password here

    text = msg.as_string()

    s.sendmail(fromaddr, toaddr, text)

    s.quit()

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["dog","leopard","human"]

# Images path
images_path = glob.glob(r"C:\Users\suraj\OneDrive\Desktop\Chetah images\*.jpg")



layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)
# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            if((str(label)=="human") or (str(label)=="leopard")):
                send_email(label)

            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)


    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()