from fastapi import FastAPI, Request
from PIL import Image
import os
import io
import cv2
import numpy as np
from datetime import datetime
import uvicorn
import base64
import onnxruntime as rt
import onnx
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build


scopes = ["https://www.googleapis.com/auth/spreadsheets", 'https://www.googleapis.com/auth/drive']
creds = Credentials.from_service_account_file("/etc/secrets/creds.json", scopes=scopes)
service = build('drive', 'v3', credentials=creds)
client = gspread.authorize(creds)
sheet_id = "1TcKkR2PkMSnJS76b_LioEbjJSOprZm4JXCIcl1WVFpg"
folder_id = "1Q9me-FIheyGveRIkqp9IdyLqK1KjQgBY"
sheet = client.open_by_key(sheet_id)
worksheet = sheet.get_worksheet(0)

app = FastAPI()

def upload_drive(formatted_time, count):
    metadata = {
        'name': formatted_time,
        'parents': [folder_id]
    }

    media = MediaFileUpload(f"{formatted_time}.jpg", mimetype="image/jpeg")
    file = service.files().create(
        body = metadata,
        media_body=media,
        fields='id'
    ).execute()
    return

def predict_count(img_init):
    h_init, w_init, _ = img_init.shape
    print(f"Size: {w_init} x {h_init} px")
    scale_x = w_init / 640
    scale_y = h_init / 640
    img = cv2.resize(img_init, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype(np.float32) / 255.0  

    ortvalue = np.array(img)
    sess = rt.InferenceSession('yolov8n.onnx', providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    results = sess.run([label_name], {input_name: img})
    output = np.transpose(np.squeeze(results))
    boxes, conf = [], []
    for pred in output:
        x, y, w, h = pred[:4]
        person_conf = pred[4]

        if person_conf >= 0.4:
            x1 = int((x - w / 2)*scale_x)
            y1 = int((y - h / 2)*scale_y)
            x2 = int((x + w / 2)*scale_x)
            y2 = int((y + h / 2)*scale_y)
            boxes.append([x1, y1, x2, y2])
            conf.append(float(person_conf))

    indices = cv2.dnn.NMSBoxes(boxes, conf, 0.4, 0.9)
    # print(len(boxes))
    # print(len(indices))

    final_boxes = []
    final_confidences = []


    for i in indices:  # Indices is already iterable
        final_boxes.append(boxes[i])
        final_confidences.append(conf[i])

    # Draw the final detections on the image
    for box, confidence in zip(final_boxes, final_confidences):
        x1, y1, x2, y2 = box
        # print(x1)

        # Draw the bounding box
        img_init = cv2.rectangle(img_init, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Add the confidence score as text
        label = f"Conf: {confidence:.2f}"  # Format confidence to 2 decimal places
        img_init = cv2.putText(img_init, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img_init, len(indices)

@app.get("/")
async def root():
    return {"Hello": "Mundo"}
    
@app.post("/predict")
async def predict(request: Request):
    image_data = await request.body()
    img = Image.open(io.BytesIO(image_data))
    img = np.array(img)
    img, count = predict_count(img)
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    worksheet.append_row([formatted_time, count], table_range='A1')
    cv2.imwrite(f"{formatted_time}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    upload_drive(formatted_time, count)
    print(f"count:{count}")
    return {"count": count}
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)