from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from loguru import logger
import time
from fastapi.middleware.cors import CORSMiddleware
from math import cos, sin, radians
from io import BytesIO
import os

# Configure loguru logging
logger.add("file_{time}.log", rotation="500 MB")  # Log to file with size-based rotation

app = FastAPI()

# Initialize YOLOv8 model
model = YOLO('runs/obb/train/weights/best.pt')

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
logger.info("PaddleOCR model initialized")

# Define CORS origins
origins = [
    "http://localhost",
    "http://localhost:8001",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    formatted_process_time = '{0:.2f}'.format(process_time * 1000)
    logger.info(f"method={request.method} path={request.url.path} status_code={response.status_code} process_time={formatted_process_time}ms")
    return response

def get_rotated_corners(x_center, y_center, width, height, angle):
    angle = radians(angle)
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    half_width = width / 2
    half_height = height / 2

    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]

    rotated_corners = []
    for x, y in corners:
        new_x = x * cos_angle - y * sin_angle + x_center
        new_y = x * sin_angle + y * cos_angle + y_center
        rotated_corners.append((new_x, new_y))

    return rotated_corners

def recognize_text_ocr(image, box):
    x_center, y_center, width, height, rotation_angle = box
    x_center = float(x_center)  # Ensure float
    y_center = float(y_center)  # Ensure float
    width = float(width)        # Ensure float
    height = float(height)      # Ensure float
    rotation_angle = float(rotation_angle)  # Ensure float
    
    corners = get_rotated_corners(x_center, y_center, width, height, rotation_angle)

    # 找到包含旋转矩形的最小边界矩形
    src_pts = np.array(corners, dtype="float32")
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # 计算仿射变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 应用仿射变换，裁剪图像
    cropped_image = cv2.warpPerspective(image, M, (int(width), int(height)))
    # 如果图像颠倒，翻转它
    if rotation_angle > 90:
        cropped_image = cv2.flip(cropped_image, -1)

    # 保存裁剪后的图像
    cv2.imwrite('cropped_image.jpg', cropped_image)

    # 使用 PaddleOCR 进行文本识别
    ocr_result = ocr_model.ocr(cropped_image, cls=True)

    if not ocr_result or not ocr_result[0]:
        return ""

    text = ""
    for line in ocr_result[0]:
        text += line[1][0] + " "
    return text.strip()

# 运行FastAPI应用
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "yolov8-obb-paddleocr Inference API"}

@app.post("/object-to-json")
async def object_to_json(file: UploadFile = File(...)):
    logger.info(f"Received file for /object-to-json: {file.filename}")
    try:
        # Load image from file
        image = Image.open(io.BytesIO(await file.read()))
        # Convert image to numpy array for processing
        img_array = np.array(image)
        # Perform inference
        results = model(img_array)
        
        detections = []
        for result in results:
            boxes = result.obb.xywhr
            labels = result.names
            confidences = result.obb.conf
            
            for i in range(len(boxes)):
                box = boxes[i]
                label = labels[int(result.obb.cls[i])]
                confidence = confidences[i]
                
                x_center, y_center, width, height, rotation_angle = box
                x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                rotation_angle = float(rotation_angle) * 180 / np.pi # Convert radians to degrees
                box = x_center, y_center, width, height, rotation_angle
                
                text = recognize_text_ocr(img_array, box)
                
                detections.append({
                    "label": label,
                    "confidence": float(confidence),
                    "box": {
                        "x_center": box[0],
                        "y_center": box[1],
                        "width": box[2],
                        "height": box[3],
                        "rotation_angle": box[4]
                    },
                    "text": text
                })
        
        logger.info(f"Detection results for {file.filename}: {detections}")
        return JSONResponse(content={"detections": detections})
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/object-to-img")
async def object_to_img(file: UploadFile = File(...)):
    logger.info(f"Received file for /object-to-img: {file.filename}")
    try:
        # Load image from file
        image = Image.open(io.BytesIO(await file.read()))
        # Convert image to numpy array for processing
        img_array = np.array(image)
        # Perform inference
        results = model(img_array)
        
        draw = ImageDraw.Draw(image)

        # Load a default font
        font_path = "simfang.ttf"
        if not os.path.exists(font_path):
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, 20)

        for result in results:
            boxes = result.obb.xywhr
            labels = result.names
            confidences = result.obb.conf
            
            for i in range(len(boxes)):
                box = boxes[i]
                label = labels[int(result.obb.cls[i])]
                confidence = confidences[i]

                x_center, y_center, width, height, rotation_angle = box
                x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                rotation_angle = float(rotation_angle) * 180 / np.pi # Convert radians to degrees
                box = x_center, y_center, width, height, rotation_angle

                text = recognize_text_ocr(img_array, box)
                
                corners = get_rotated_corners(x_center, y_center, width, height, rotation_angle)
                corners = [(int(x), int(y)) for x, y in corners]
                
                # Draw the bounding box
                draw.polygon(corners, outline="blue", width=6)
                
                # Calculate the text position below the bottom edge of the bounding box
                bottom_edge_midpoint_x = (corners[2][0] + corners[3][0]) / 2
                bottom_edge_midpoint_y = (corners[2][1] + corners[3][1]) / 2
                text_position = (bottom_edge_midpoint_x, bottom_edge_midpoint_y - 10)  # 10 pixels below the bottom edge
                
                # Draw the label and text
                draw.text(text_position, f"{label}: {text}", fill="red", font=font, anchor="ms")  # "ms" means middle of the text should be positioned at the given coordinates

        # Convert image to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

