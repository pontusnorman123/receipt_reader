import os
from PIL import Image
import requests
import json
import time

# Directory containing images
image_folder = 'image_folder'
output_data = []
server_url = 'http://localhost:5000/process'  # Adjust this to your server's address

def send_image_to_server(image_path, retries=5, delay=2):
    """Sends an image to the server and tries multiple times if it fails."""
    attempt = 0
    while attempt < retries:
        try:
            with open(image_path, 'rb') as img:
                files = {'image': img}
                response = requests.post(server_url, files=files)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Failed to process {image_path}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
        time.sleep(delay)  # Wait before retrying
        attempt += 1
    raise Exception(f"Failed to send image after {retries} attempts")

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        try:
            result = send_image_to_server(image_path)
            processed_entry = {'image_name': filename, 'data': result}
            output_data.append(processed_entry)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Save all results to a JSON file
with open('results/processed_receipts_data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
