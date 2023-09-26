import cv2
import time
import numpy as np

# Read output gif
cap = cv2.VideoCapture("output/movie.gif")
buffer = []
max_val_array = np.empty(0)
sampleCount = 0

# While Video is active
while True:
    # Read Image
    ret, frame = cap.read()

    # Slow down video for visualization
    time.sleep(0.1) # 10 FPS

    # If video stream done, then end
    if not ret:
        break

    if len(buffer) == 0:
        # If buffer not yet populated, populate buffer
        buffer.append(frame)
        sample_count = 1 # initialization
        mean_val = 1 
        std_dev = 1
        max_val_array = np.append(max_val_array, 1)
        print("Init")
        continue
    elif len(buffer) == 1:
        print("Re-allocate")
        buffer.append(frame)
    else:
        # Update sliding buffer to calculate similarities    
        buffer[0] = buffer[-1]
        buffer[1] = frame
    

    sampleCount += 1
    res = cv2.matchTemplate(buffer[0], buffer[1], cv2.TM_CCORR_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    max_val_array = np.append(max_val_array, max_val)
    # Simplified std_dev equation when N = n - 1, with n = 2
    diff = abs(max_val - mean_val)
    print((max_val, mean_val, sampleCount, diff, std_dev))

    if (diff > 3*std_dev): # If difference outside of std dev, filter out
        print("Skipping Frame")
        # Remove newest frame from buffer to not skew results
        buffer.pop(1)
        max_val_array = np.delete(max_val_array, -1)
        cv2.imshow('skipped_frame', frame)
        # Wait for keypress
        if cv2.waitKey(1) == ord('q'):
            break
        continue # Skip Image
    else:
        # Update mean and std dev if within range
        mean_val = np.mean(max_val_array)
        std_dev = np.std(max_val_array)
        # Show Raw Image
        cv2.imshow('frame', frame)
        # Wait for keypress
        if cv2.waitKey(1) == ord('q'):
            break

# Clean-up resources
cap.release()
cv2.destroyAllWindows()
