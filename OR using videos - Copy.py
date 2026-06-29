
import numpy as np
import imutils  # resize the image
import cv2
import time

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.5  # Confidence threshold


# CHANGE THIS TO YOUR VIDEO FILE PATH
video_path = "OR video.mp4"  
save_output_video = True 
output_filename = "output_detections.mp4"  # Output file name



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor", "mobile"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print(" Model Loaded Successfully")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

print(f"Opening video: {video_path}")
vs = cv2.VideoCapture(video_path)

if not vs.isOpened():
    print("Error: Cannot open video file")
    exit()

# Get video properties
fps = vs.get(cv2.CAP_PROP_FPS)
total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n--- Video Properties ---")
print(f"FPS: {fps:.2f}")
print(f"Total Frames: {total_frames}")
print(f"Resolution: {frame_width}x{frame_height}")
print(f"Duration: {total_frames/fps:.2f} seconds")

# Setup output video writer
output_writer = None
if save_output_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be saved to: {output_filename}")

print("\n--- Starting Detection ---\n")

frame_count = 0
fps_counter_time = time.time()
start_time = time.time()
detection_count = 0
current_fps = 0

while True:
    ret, frame = vs.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Resize frame for processing
    frame_resized = imutils.resize(frame, width=1000)
    (h, w) = frame_resized.shape[:2]
    
    # Create blob
    blob = cv2.dnn.blobFromImage(frame_resized, 
        0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    detShape = detections.shape[2]
    current_detections = 0
    
    for i in np.arange(0, detShape):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            
            if idx < len(CLASSES):
                current_detections += 1
                detection_count += 1
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                
                cv2.rectangle(frame_resized, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                
                if startY - 15 > 15:
                    y = startY - 15
                else:
                    y = startY + 15
                
                cv2.putText(frame_resized, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    # Calculate FPS every 30 frames
    if frame_count % 30 == 0:
        elapsed = time.time() - fps_counter_time
        current_fps = 30 / elapsed
        fps_counter_time = time.time()
    
    # Progress overlay
    progress_percent = (frame_count / total_frames) * 100
    info_text = f"Frame: {frame_count}/{total_frames} ({progress_percent:.1f}%)"
    cv2.putText(frame_resized, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if current_fps > 0:
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame_resized, fps_text, (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if current_detections > 0:
        det_text = f"Detections: {current_detections}"
        cv2.putText(frame_resized, det_text, (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Write to output video
    if output_writer is not None:
        frame_output = cv2.resize(frame_resized, (frame_width, frame_height))
        output_writer.write(frame_output)
    
    # Display
    cv2.imshow("Video Object Detection", frame_resized)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nStopped by user")
        break

# Summary
total_time = time.time() - start_time
avg_fps = total_frames / total_time if total_time > 0 else 0

print(f"\n--- Detection Summary ---")
print(f"Total Detections: {detection_count}")
print(f"Total Processing Time: {total_time:.2f} seconds")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Average Detections per Frame: {detection_count/total_frames:.2f}")

# Cleanup
vs.release()
if output_writer is not None:
    output_writer.release()
    print(f"\n Output video saved: {output_filename}")
cv2.destroyAllWindows()
print("Done!")



