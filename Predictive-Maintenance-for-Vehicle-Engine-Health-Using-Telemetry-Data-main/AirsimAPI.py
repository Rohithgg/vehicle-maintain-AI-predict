import airsim
import time
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pynput import keyboard  # For keyboard control

# Connect to AirSim
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# Load trained AI model
model = load_model("bike_health_model.keras")

# Define car controls
car_controls = airsim.CarControls()

# Define default control values
throttle = 0.0
steering = 0.0
brake = 0.0

# Initialize buffer for AI input
sequence_length = 5
buffer = []

# Initialize CSV log file
csv_file = "predict_log.csv"
columns = ["Speed", "EngineTemperature", "FuelLevel"]
df = pd.DataFrame(columns=columns)
df.to_csv(csv_file, index=False)

# Keyboard event handlers
def on_press(key):
    global throttle, steering, brake
    try:
        if key.char == 'w':  # Accelerate
            throttle = min(throttle + 0.1, 1.0)
        elif key.char == 's':  # Brake/Reverse
            brake = min(brake + 0.1, 1.0)
        elif key.char == 'a':  # Steer Left
            steering = max(steering - 0.1, -1.0)
        elif key.char == 'd':  # Steer Right
            steering = min(steering + 0.1, 1.0)
    except AttributeError:
        pass

def on_release(key):
    global throttle, steering, brake
    if key == keyboard.Key.esc:
        return False  # Stop listener when ESC is pressed

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Simulation loop
while True:
    try:
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get car state
        car_state = client.getCarState()

        # Simulated values
        speed = car_state.speed  # Speed in m/s
        rpm = speed * 100  # Simplified RPM calculation
        fuel_level = max(100 - (speed * 0.1), 0)  # Simulated fuel consumption

        # Store last 5 readings for AI model
        buffer.append([speed, rpm, fuel_level])
        if len(buffer) > sequence_length:
            buffer.pop(0)

        # AI Model Prediction (When buffer is filled)
        if len(buffer) == sequence_length:
            input_data = np.array([buffer])  # Convert to NumPy array
            prediction = model.predict(input_data)

            # AI-predicted values
            speed_pred, rpm_pred, fuel_pred = prediction[0]

            # Print AI predictions
            print(f"🚀 AI Predictions - Speed: {speed_pred:.2f}, RPM: {rpm_pred:.2f}, Fuel: {fuel_pred:.2f}")

            # Save predictions to CSV
            df = pd.DataFrame([[timestamp, speed, speed_pred, rpm, rpm_pred, fuel_level, fuel_pred]], columns=columns)
            df.to_csv(csv_file, mode='a', header=False, index=False)

        # Apply user controls
        car_controls.throttle = throttle
        car_controls.steering = steering
        car_controls.brake = brake
        client.setCarControls(car_controls)

        # Wait before next reading
        time.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")
        break
