# server to read live data from the UNO
# the UNO will read from the sensors and send the data to this server

import serial
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained AI model
model = load_model("bike_health_model.h5")  # Make sure to save your trained model first

# Connect to Arduino Serial Port (change COM3 to your port)
arduino = serial.Serial(port="COM3", baudrate=9600, timeout=1)

# Dummy buffer for real-time sequence data (last 5 readings)
sequence_length = 5
buffer = []

while True:
    try:
        # Read Serial data from Arduino
        line = arduino.readline().decode("utf-8").strip()
        if not line:
            continue

        # Parse values (assuming format: "RPM,Temperature,FuelLevel")
        values = line.split(",")
        if len(values) != 3:
            continue  # Skip if data format is incorrect

        rpm = float(values[0])
        temperature = float(values[1])
        fuel_level = float(values[2])

        # Store in buffer (keep only last 5 readings)
        buffer.append([rpm, temperature, fuel_level])
        if len(buffer) > sequence_length:
            buffer.pop(0)

        # Make a prediction when we have enough readings
        if len(buffer) == sequence_length:
            input_data = np.array([buffer])  # Convert to NumPy array
            prediction = model.predict(input_data)

            # Example AI prediction logic
            speed_pred, temp_pred, fuel_pred = prediction[0]

            print(f"🚀 AI Predictions - Speed: {speed_pred:.2f}, Temp: {temp_pred:.2f}, Fuel: {fuel_pred:.2f}")

            # Check for possible failures
            if temp_pred > 90:
                print("⚠️ WARNING: Engine Overheating Detected!")# stop and cold down the engine
            if fuel_pred < 10:
                print("⚠️ WARNING: Low Fuel Level!") # Refuel required
            if speed_pred > 120:
                print("⚠️ WARNING: Over-speeding detected!") # Slow down

    except Exception as e:
        print(f"Error: {e}")
