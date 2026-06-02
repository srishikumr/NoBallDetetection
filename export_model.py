from ultralytics import YOLO

# 1. Load your ALREADY TRAINED PyTorch model.
# (Check your explorer to see if it's in train2 or train3)
model = YOLO('runs/classify/train3/weights/best.pt')

# 2. Export to ONNX format
print("Exporting model to ONNX...")
model.export(format='onnx')

print("Done! Your .onnx file is ready for the Raspberry Pi.")