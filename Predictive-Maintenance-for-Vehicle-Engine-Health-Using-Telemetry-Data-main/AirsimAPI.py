import airsim
import tensorflow as tf
import numpy as np
import cv2
import time

# ---------------------------
# Load the trained model
# ---------------------------
# Replace 'path_to_your_model.h5' with the path to your saved model.
model = tf.keras.models.load_model('bike_health_model.keras')

# ---------------------------
# Connect to AirSim and enable control
# ---------------------------
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# ---------------------------
# weather
# airsim.simEnableWeather(True)
# ---------------------------
# Main control loop
# ---------------------------
while True:
    try:
        # ---- Get Image Data ----
        # Request a scene image from camera "0" (adjust camera ID and type as needed)
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        if len(responses) == 0:
            print("No image received.")
            continue

        response = responses[0]
        # Convert raw image data to a numpy array and reshape
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        # Resize to the expected input size (e.g., 224x224)
        img_resized = cv2.resize(img_rgb, (224, 224))
        # Normalize pixel values to [0,1] if your model was trained that way
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Expand dimensions to add the batch dimension
        image_input = np.expand_dims(img_normalized, axis=0)

        # ---- Get Telemetry Data ----
        car_state = client.getCarState()
        # Example: Create a telemetry vector with 10 features. Adjust extraction to match your model.
        telemetry_vector = np.array([
            car_state.speed,
            car_state.kinematics_estimated.linear_acceleration.x_val,
            car_state.kinematics_estimated.linear_acceleration.y_val,
            car_state.kinematics_estimated.linear_acceleration.z_val,
            car_state.kinematics_estimated.orientation.x_val,
            car_state.kinematics_estimated.orientation.y_val,
            car_state.kinematics_estimated.orientation.z_val,
            car_state.kinematics_estimated.orientation.w_val,
            car_state.position.x_val,
            car_state.position.y_val
        ])
        telemetry_input = telemetry_vector.reshape(1, -1)

        # ---- Make Predictions ----
        # The model returns two outputs: driving commands and maintenance alert.
        predictions = model.predict([image_input, telemetry_input])
        driving_preds = predictions[0][0]  # Expected order: [steering, throttle, brake]
        maintenance_pred = predictions[1][0][0]  # Probability of requiring maintenance

        # ---- Apply Predictions to Car Controls ----
        car_controls.steering = float(driving_preds[0])
        car_controls.throttle = float(driving_preds[1])

        # For braking, you may want to override throttle if the brake prediction is high.
        brake_value = float(driving_preds[2])
        if brake_value > 0.5:  # Threshold can be adjusted based on your training.
            car_controls.brake = brake_value
        else:
            car_controls.brake = 0.0

        # Send the control commands to the simulator.
        client.setCarControls(car_controls)
        # collision reset
        if car_state.collision.has_collided:
            client.reset()
            time.sleep(1)

        # ---- Predictive Maintenance Alert ----
        if maintenance_pred > 0.5:
            print("Maintenance Alert: Vehicle may require service soon.")

        # Adjust the loop delay as needed (e.g., 100ms per iteration).
        time.sleep(0.1)

    except Exception as e:
        print("Error in control loop:", e)
        break

