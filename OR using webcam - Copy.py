import numpy as np
import imutils  # resize the image
import cv2
import time

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.5  # Increased from 0.2 to reduce false positives
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor", "mobile"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("Model Loaded Successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("Starting Camera Feed...")
vs = cv2.VideoCapture(0)

# Check if camera opened successfully
if not vs.isOpened():
    print("Error: Cannot access camera")
    exit()

time.sleep(2.0)
fps_time = time.time()
frame_count = 0

while True:
    ret, frame = vs.read()
    
    if not ret:
        print("Error reading frame")
        break
    
    # Resize frame for processing
    frame = imutils.resize(frame, width=1000)  # Reduced from 1000 for faster processing
    (h, w) = frame.shape[:2]
    
    # Create blob with proper normalization for MobileNetSSD
    # Scale: 1/127.5, Size: 300x300, Mean subtraction: 127.5
    blob = cv2.dnn.blobFromImage(frame, 
        0.007843,           # scale factor (1/127.5)
        (300, 300),         # size
        127.5)              # mean subtraction
    
    net.setInput(blob)
    detections = net.forward()
    
    # Get detections
    detShape = detections.shape[2]
    
    for i in np.arange(0, detShape):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confThresh:
            # Get class index
            idx = int(detections[0, 0, i, 1])
            
            # Ensure idx is valid
            if idx < len(CLASSES):
                print(f"Detected: {CLASSES[idx]} | Confidence: {confidence:.2f}")
                
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within frame bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Create label
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                
                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                
                # Calculate text position (FIXED BUG)
                if startY - 15 > 15:
                    y = startY - 15
                else:
                    y = startY + 15  # Fixed: was 'startY + 15' without assignment
                
                # Draw text label
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    # Calculate and display FPS
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed_time = time.time() - fps_time
        fps = 30 / elapsed_time
        print(f"FPS: {fps:.2f}")
        fps_time = time.time()
    
    # Display frame
    cv2.imshow("MobileNetSSD Detection", frame)
    
    # Exit on ESC key
    key = cv2.waitKey(1)
    if key == 27:
        print("Exiting...")
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()
