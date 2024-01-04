import io
import os
import cv2
import uvicorn
import cvlib as cv
from cvlib.object_detection import draw_bbox
#import nest_asyncio
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import requests
from io import BytesIO
import subprocess
import json
import uuid
from PIL import Image, ImageDraw, ImageFont

# Create the images, jsons and annotated directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")
    
if not os.path.exists("jsons"):
    os.makedirs("jsons")

if not os.path.exists("annotated"):
    os.makedirs("annotated")
            
app = FastAPI(title='How good is your Object Detection Model?')

@app.get("/")
def home():
    return "Head over to http://localhost:8000/docs."


@app.post("/json_prediction")
async def json_prediction(file: UploadFile = File(...)):
    
    # Generate a unique ID for the image
    unique_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    image_filename = f"{unique_id}.{file_extension}"


    # Save the uploaded file locally
    with open(f'images/{image_filename}', "wb") as image_file:
        content = await file.read()
        image_file.write(content)
    
    # Execute the curl command
    result = subprocess.run(
        ["curl", "http://torchserve:8080/predictions/fasterrcnn_resnet50", "-T", f'images/{image_filename}'],
        capture_output=True,
        text=True,
        check=True
        )
        
    # Parse the output string as JSON
    output_json = json.loads(result.stdout)

    # Add the unique ID to the JSON output
    response = {
        "image_id": unique_id,
        "detected_objects": output_json
    }
    
    # Save the JSON response to a file
    with open(f"jsons/{unique_id}.json", "w") as json_file:
        json.dump(response, json_file)

    return response

@app.get("/annotated_image/{image_id}")
async def get_annotated_image(image_id: str, threshold: float = 0.5):

    # Load json output
    file_path = f"jsons/{image_id}.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Load the original image
    image_path = f"images/{image_id}.jpg"  # Replace with your image path
    original_image = Image.open(image_path)

    # Get the detected objects for the requested image ID
    objects = data["detected_objects"]

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(original_image)
    for obj in objects:
        for label, coords in obj.items():
            if isinstance(coords, list) and len(coords) == 4 and obj["score"] >= threshold:
                # Convert floating-point coordinates to integers
                coordinates = [int(coord) for coord in coords]
                draw.rectangle(coordinates, outline="red", width=1)
                draw.text((coordinates[0], coordinates[1]), label, fill="red", font = ImageFont.load_default().font_variant(size=12))


    # Save or serve the annotated image
    annotated_image_path = f"annotated/{image_id}.jpg"  # Save the annotated image with a unique name
    original_image.save(annotated_image_path)

    return FileResponse(annotated_image_path, media_type="image/jpeg")