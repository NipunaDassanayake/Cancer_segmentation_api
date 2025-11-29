from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
app = Flask(__name__)
CORS(app)

# --- Model Parameters ---
MODEL_FILE = 'yolov8n-seg.pt'
CONFIDENCE_THRESHOLD = 0.40  # Reverting to a low, stable threshold
IOU_THRESHOLD = 0.7
IMG_SIZE = 256
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]

# --- 2. GLOBAL MODEL LOADING ---
try:
    MODEL = YOLO(MODEL_FILE)
    print(f"✅ Model loaded successfully: {MODEL_FILE}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model. Ensure {MODEL_FILE} is present.")
    print(e)


# --- 3. UTILITY FUNCTION FOR INFERENCE ---
def get_segmentation_base64(image_bytes):
    """Processes image, runs YOLOv8 segmentation, and returns the multi-colored result."""

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_to_draw = img_rgb.copy()

    # 2. Run Prediction with the set threshold (0.40)
    results_list = MODEL.predict(
        source=img_rgb,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )
    results = results_list[0]

    tumors_detected = 0

    # 3. Manual Multi-Color Drawing and Post-Processing
    if results.masks:
        tumors_detected = len(results.masks.data)  # ⭐ This is the correct count
        for i, mask_tensor in enumerate(results.masks.data):
            current_color = COLORS[i % len(COLORS)]

            # Resize mask to the original image dimensions
            mask_np = cv2.resize(mask_tensor.cpu().numpy().astype(np.uint8),
                                 (img_to_draw.shape[1], img_to_draw.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0

            # Apply colored overlay with transparency
            overlay = img_to_draw.copy()
            overlay[mask_np] = current_color
            img_to_draw = cv2.addWeighted(overlay, 0.4, img_to_draw, 0.6, 0)

            # Draw bounding box and label
            if results.boxes and len(results.boxes.xyxy) > i:
                box_coords = results.boxes.xyxy[i].cpu().tolist()
                x1, y1, x2, y2 = [int(c) for c in box_coords]
                conf = results.boxes.conf[i].cpu().numpy()
                label = f"tumor {conf:.2f}"

                cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), current_color, 2)
                cv2.putText(img_to_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 1,
                            cv2.LINE_AA)
    # 4. Encode the resulting image
    img_bgr_result = cv2.cvtColor(img_to_draw, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr_result)
    return base64.b64encode(buffer).decode('utf-8'), tumors_detected


# --- 4. API ENDPOINT DEFINITION ---
@app.route('/predict_segmentation', methods=['POST'])
def predict_endpoint():
    """API endpoint: Accepts a POST request with an image file."""

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded. Use form-data field "image"'}), 400

    image_bytes = request.files['image'].read()

    try:
        # Get both the image data and the correct count from the utility function
        segmented_image_base64, tumors_detected = get_segmentation_base64(image_bytes)

        return jsonify({
            'status': 'success',
            'segmented_image_png_base64': segmented_image_base64,
            'tumors_detected': tumors_detected,  # ⭐ Uses the correctly retrieved count
            'message': 'Segmentation prediction complete.'
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'Internal server error during prediction: {str(e)}'}), 500


# --- 5. RUN THE FLASK APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)