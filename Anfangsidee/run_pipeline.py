import os
import cv2
import glob
import torch
import base64
import requests
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading TrOCR model...")
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
print("Model loaded successfully.")

# 2. Vision Pipeline
def preprocess_and_segment(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None, []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_images = []
    # Sort contours roughly from top to bottom
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            crop = img[y:y+h, x:x+w]
            box_images.append(crop)
            
    return img, box_images

# 3. HTR (Handwritten Text Recognition)
def recognize_handwriting(image_crop):
    image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=64)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

# 4. Mermaid Generation
def generate_mermaid_code(diagram_type, texts):
    if diagram_type == "flow-chart":
        mermaid_code = "graph TD;\\n"
        for i, text in enumerate(texts):
            clean_text = text.replace('"', '').replace(';', '')
            mermaid_code += f'    Node{i}["{clean_text}"]\\n'
            # Sequential connection
            if i > 0:
                mermaid_code += f'    Node{i-1} --> Node{i}\\n'
                
    elif diagram_type == "state-diagram":
        mermaid_code = "stateDiagram-v2\\n"
        mermaid_code += "    [*] --> State0\\n"
        for i, text in enumerate(texts):
            clean_text = text.replace('"', '').replace(';', '')
            mermaid_code += f'    State{i} : {clean_text}\\n'
            if i > 0:
                mermaid_code += f'    State{i-1} --> State{i}\\n'
        mermaid_code += f'    State{len(texts)-1} --> [*]\\n'
    else:
        mermaid_code = "graph TD;\\n    A[Unknown Diagram Type]"
        
    return mermaid_code

def render_mermaid_to_image(mermaid_code, output_path):
    b64 = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{b64}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

# 5. Main Execution Area, should work now
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

sketch_dirs = glob.glob("sketch-data/*")
num_processed = 0

for d in sketch_dirs:
    if not os.path.isdir(d): continue
    
    diagram_type = os.path.basename(d)
    print(f"\\n=== Processing directory: {diagram_type} ===")
    
    images = glob.glob(os.path.join(d, "*.jpg"))
    for img_path in images: # Process up to 2 images for quick test
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\\nProcessing {img_path}...")
        
        vis_img, crops = preprocess_and_segment(img_path)
        if not crops:
            print("No boxes found.")
            continue
            
        print(f"Found {len(crops)} boxes.")
        
        recognized_texts = []
        for i, crop in enumerate(crops):
            text = recognize_handwriting(crop)
            recognized_texts.append(text)
            print(f"  Box {i}: {text}")
            
        mermaid_code = generate_mermaid_code(diagram_type, recognized_texts)
        
        mmd_path = os.path.join(output_dir, f"{diagram_type}_{base_name}.mmd")
        with open(mmd_path, "w") as f:
            f.write(mermaid_code)
            
        png_path = os.path.join(output_dir, f"{diagram_type}_{base_name}.png")
        if render_mermaid_to_image(mermaid_code, png_path):
            print(f"Saved PNG to {png_path}")
        else:
            print(f"Failed to generate PNG for {base_name}")
            
        num_processed += 1
        if num_processed >= 2: # Keep the test short
            break
    if num_processed >= 2:
        break
        
print(f"\\nFinished test. Processed {num_processed} images.")
