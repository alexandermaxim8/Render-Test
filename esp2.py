import requests
import json
import cv2
import base64

url = "https://render-test-1-pum2.onrender.com"

with open("Commuters board an EDSA_.jpg", "rb") as img_file:
    image_data = img_file.read()
headers = {"Content-Type": "image/jpeg"}
response = requests.post(f"{url}/predict", data=image_data, headers=headers)
print(response.status_code)
print("response:", response.json())