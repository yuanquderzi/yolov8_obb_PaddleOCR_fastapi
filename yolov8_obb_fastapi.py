from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load YOLOv8 model
model = YOLO('runs/obb/train/weights/best.pt')

@app.post("/object-to-json")
async def object_to_json(file: UploadFile = File(...)):
    # Load image from file
    image = Image.open(io.BytesIO(await file.read()))
    # Perform inference
    results = model(image)
    
    # Extract results
    detections = []
    for result in results:
        # Extracting boxes, labels, and confidences
        boxes = result.obb.xywhr
        labels = result.names  # Assuming `result.names` contains class names
        confidences = result.obb.conf  # Confidence scores
        
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[int(result.obb.cls[i])]  # Class name
            confidence = confidences[i]
            detections.append({
                "label": label,
                "confidence": float(confidence),
                "box": {
                    "x_center": float(box[0]),
                    "y_center": float(box[1]),
                    "width": float(box[2]),
                    "height": float(box[3]),
                    "rotation_angle": float(box[4]) * 180 / np.pi  # Convert radians to degrees
                }
            })
    
    return JSONResponse(content={"detections": detections})

# 定义 /object-to-img 接口
@app.post("/object-to-img")
async def object_to_img(file: UploadFile = File(...)):
    # 将上传的文件数据读取到内存中
    image_data = await file.read()
    
    # 打开图像并转换为适当的格式
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 使用模型进行预测
    results = model(image)
    
    # 获取处理后的图像数据
    img_byte_arr = BytesIO()
    for result in results:
        # 将 numpy.ndarray 转换为 PIL.Image
        annotated_image = Image.fromarray(result.plot())
        annotated_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

# Run the app using: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

