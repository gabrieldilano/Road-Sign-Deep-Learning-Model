import requests

# Define file path
file_path = r"C:\Users\crisp\Documents\ML\Road Sign Model\Road-Sign-Deep-Learning-Model\stop-sign-x-r1-1_pl.png"

with open(file_path, 'rb') as f:
    # Prepare the file to be sent in request
    files = {'file': f}
    
    # Send POST request to Flask API
    response = requests.post('http://127.0.0.1:5000/predict', files=files)

print(response.json())
