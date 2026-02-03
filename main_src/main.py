from keras.models import load_model   # TensorFlow Keras model loading for compatibility
import cv2      # OpenCV for video capture and image processing
import numpy as np  # NumPy for numerical operations
from collections import deque  # Deque for maintaining prediction history

def camera_recognition():
    np.set_printoptions(suppress=True)      # Suppress scientific notation in NumPy printouts

    # Load the pre-trained Keras model
    model = load_model("converted_keras/keras_Model.h5", compile=False)

    # Load class names from labels file
    class_names = [line.strip() for line in open("converted_keras/labels.txt")]

    # Initialize webcam
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize prediction history deque
    pred_history = deque(maxlen=5)
    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            continue
        frame_count += 1
        # Preprocess the frame
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        # Perform prediction every 3 frames
        if frame_count % 3 == 0:
            image = np.asarray(resized, dtype=np.float32)
            image = (image / 127.5) - 1
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image, verbose=0)
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            if confidence > 0.7:
                pred_history.append(index)
        # Determine the most frequent prediction in history
        if len(pred_history) > 0:
            final_index = max(set(pred_history), key=pred_history.count)
            label = class_names[final_index]
            conf_percent = int(confidence * 100)
        else:
            label = "Unknown"
            conf_percent = 0
        # Overlay the prediction on the frame
        cv2.putText(
            frame,
            f"{label} {conf_percent}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        # Display the resulting frame
        cv2.imshow("Webcam Recognition", frame)
        if cv2.waitKey(1) == 27: # Exit on 'ESC' key
            break
    camera.release()
    cv2.destroyAllWindows()