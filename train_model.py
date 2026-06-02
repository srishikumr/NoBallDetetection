from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 Nano Classification model
# Nano ('n') is the smallest and fastest model, perfect for Edge devices.
model = YOLO('yolov8n-cls.pt')

# 2. Train the model on your dataset
# Replace 'my_cricket_dataset' with the actual path to your folder
print("Starting training...")
results = model.train(
    data='my_cricket_dataset', 
    epochs=25,       # Number of times it loops through your dataset
    imgsz=224,       # Resizes images to be smaller and process faster
    device='cpu'     # Change to '0' if you have an Nvidia GPU setup
)

# 3. Export for the Raspberry Pi
# The int8 quantization compresses the model so it runs fast on small processors.
print("Exporting model for Edge computing...")
# model.export(format='tflite', int8=True)

# print("Done! Look in the 'runs/classify/train/weights/' folder for your .tflite model.")