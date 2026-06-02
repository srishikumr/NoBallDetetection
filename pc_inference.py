import cv2
from ultralytics import YOLO

# 1. Load your trained PyTorch model
model_path = 'runs/classify/train3/weights/best.pt' 
model = YOLO(model_path)

# 2. Read the static image file directly
image_path = "7 copy.jpg"
frame = cv2.imread(image_path)

# Check if the image loaded successfully before processing
if frame is None:
    print(f"Error: Could not find '{image_path}'. Make sure it's in this folder!")
else:
    print("Running analytics on your image...")
    
    # 3. Pass the static image frame straight to the model
    results = model(frame, verbose=False)

    # 4. Extract the prediction class and confidence percentage
    top_class_index = results[0].probs.top1
    predicted_class = results[0].names[top_class_index]
    confidence = results[0].probs.top1conf.item() * 100

    # 5. Set text color based on result (Green for legal, Red for no ball)
    if predicted_class == "legal":
        color = (0, 255, 0)
        text = f"LEGAL:"
    else:
        color = (0, 0, 255)
        text = f"NO BALL WARNING:"

    # 6. Overlap the text prediction onto the upper corner of the image
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # 7. Open a display window to show you the final analyzed picture
    cv2.imshow("Static Cricket Analytics", frame)

    # This keeps the image open on your screen until you press ANY key
    cv2.waitKey(0) 
    cv2.destroyAllWindows()