import cv2
from Face_Site import Predict

def test_model_from_camera():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Use the Predict function to process the frame
        processed_frame = Predict(frame)
        
        # Display the processed frame
        cv2.imshow('Emotion Detection', processed_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Test the model with the webcam
test_model_from_camera()