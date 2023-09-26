## This file is an implementation of a filter which rejects video frames that don't meet ANY of the following criteria:
## 1. Does not meet the histogram similarity threshold
##    - Histograms are caculated after converting images into HSV (Hue-Saturation-Value) space. Bins are created in HSV space for simplicity compared to RGB
## 2. Does not meet the dynmaically calculated std deviation threshold multiplied by a sigma multiplier.
##    - 3 is selected as the sigma multiplier because an assumption of the normal guassian noise distribution of 3 sigma is used
## Optionally bilateral filtering and histogram equalization can be enabled for smoother video output. It is enabled by default


import cv2
import time
import numpy as np
import imageio

# Set buffer array, max value array for calculating stats, and flag to determine whether filter is init
buffer = []
filtered_images = []
max_val_array = np.empty(0)
is_init = False
fps = 15

# Algorithm Parameters
histogram_similarity_threshold = 0.95
sigma_multiplier = 3.0
h_bins = 50
s_bins = 60
h_ranges = [0, 180]
s_ranges = [0, 256]
channels = [0, 1]
enable_bilateral_and_histogram = True

# Read output gif from UnLiteFlowNetPIV
cap = cv2.VideoCapture("output/raw_movie.gif")

# While Video is active
while True:
    # Read Image
    ret, frame = cap.read()

    # Slow down video for visualization
    time.sleep(0.1) # 10 FPS

    # If video stream done, then end
    if not ret:
        break

    # If buffer not yet populated, populate buffer. Assumes first image in set is NOT bad
    if len(buffer) == 0:
        buffer.append(frame)
        mean_val = 1 
        std_dev = 0.1
        max_val_array = np.append(max_val_array, 1)
        print("Init")
        continue
    # If buffer is not full, append it with new frame
    elif len(buffer) == 1:
        buffer.append(frame)
    else:
    # If buffer is full, swap the previous new frame to best the current old frame. Update new frame to be template
        buffer[0] = buffer[-1]
        buffer[1] = frame
    
    # Convert RGB to HSV space for histogram processing
    hsv_0 = cv2.cvtColor(buffer[0], cv2.COLOR_BGR2HSV)
    hsv_1 = cv2.cvtColor(buffer[1], cv2.COLOR_BGR2HSV)

    # Condition 1: Histogram comparison in HSV color space using cross correlation
    histSize = [h_bins, s_bins]

    ranges = h_ranges + s_ranges

    hist_0 = cv2.calcHist([hsv_0], channels, None, histSize, ranges, accumulate=False)
    hist_1 = cv2.calcHist([hsv_1], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_0, hist_0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    compare_val = cv2.compareHist(hist_0, hist_1, cv2.HISTCMP_CORREL)

    # Condition 2: Cross Correlation Template Matching Normalized
    res = cv2.matchTemplate(hsv_0, hsv_1, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    max_val_array = np.append(max_val_array, max_val)

    # Simplified std_dev equation when N = n - 1, with n = 2
    diff = abs(max_val - mean_val)

    # Print current status
    print(str(('Template_Matching_Value', 'Total Mean', 'Frame std_dev', 'Total std_dev', 'histogram similarity')) + " : " + str((max_val, mean_val, diff, std_dev, compare_val)))

    # Check if ANY of the filter criteria is met
    if (diff > sigma_multiplier*std_dev) or (compare_val < histogram_similarity_threshold): 
        # If filter criteria is not met, and the filter is not initialized, try a new first frame.
        if is_init is False:
            print("Re-init, first frame was bad")
            buffer.pop(-1)
            buffer.pop(-1)
        # Skip this frame, it is being filtered, reset buffer and mean calculation to not skew data
        else:
            print("Skipping Frame")
            # Remove newest frame from buffer to not skew results
            buffer.pop(-1)
       
            # Clean up statistics array to not skew data
            max_val_array = np.delete(max_val_array, -1)

            # Show skipped frame in different window
            cv2.imshow('skipped_frame', frame)

            # Wait for keypress
            if cv2.waitKey(1) == ord('q'):
                break

            # Skip any further processing and go to next iteration / frame
            continue 
    # Filter criteria not met, update mean and std_dev calculation 
    else:
        # Mark the filter as initialized if not already
        if is_init is False:
            is_init = True

        # Update mean and std dev for next iteration 
        mean_val = np.mean(max_val_array)
        std_dev = np.std(max_val_array)

        # Show Raw Image
        cv2.imshow('frame', frame)

        # Collect Filtered Images and apply histogram equalization + bilateral filtering if selected
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if enable_bilateral_and_histogram is True:
            hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
            hsv_frame[:, :, 2] = cv2.equalizeHist(hsv_frame[:,:,2])
            rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
            rgb_frame = cv2.bilateralFilter(rgb_frame, 9, 75, 75)
     
        filtered_images.append(rgb_frame)

        # Wait for keypress
        if cv2.waitKey(1) == ord('q'):
            break


# Write GIF
print("Writing output filtered gif to output/filtered_movie.guf")
imageio.mimsave('output/filtered_movie.gif', filtered_images, format='GIF', duration=1000/int(fps))

# Clean-up resources
cap.release()
cv2.destroyAllWindows()
print("Done")
