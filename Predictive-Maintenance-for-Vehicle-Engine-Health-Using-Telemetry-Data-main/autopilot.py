import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Define the Input Layers

# Input for camera images (e.g., 224x224 RGB image)
image_input = Input(shape=(224, 224, 3), name='image_input')

# Input for vehicle telemetry data (e.g., speed, engine temp, etc.)
# Adjust the number of telemetry features as needed.
telemetry_input = Input(shape=(10,), name='telemetry_input')

# Build the Image Processing Branch (CNN)

x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(image_input)
x = layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
x = layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
image_features = layers.Dense(100, activation='relu')(x)

# Build the Telemetry Processing Branch (Fully Connected)

y = layers.Dense(50, activation='relu')(telemetry_input)
y = layers.Dense(50, activation='relu')(y)

# Combine Features from Both Branches

combined = layers.Concatenate()([image_features, y])
combined = layers.Dense(100, activation='relu')(combined)


# Output Branch 1: Driving Commands
# (For example, outputting steering, throttle, and brake values)

driving_output = layers.Dense(3, activation='linear', name='driving_output')(combined)


# Output Branch 2: Predictive Maintenance / Vehicle Health Monitoring
# (For example, a binary indicator: 0 = healthy, 1 = requires maintenance)
# You could also design this branch for regression if you predict a health score.

maintenance_output = layers.Dense(1, activation='sigmoid', name='maintenance_output')(combined)


# Build and Compile the Model

model = Model(inputs=[image_input, telemetry_input],
              outputs=[driving_output, maintenance_output])

model.compile(
    optimizer='adam',
    loss={
        'driving_output': 'mse',              # Regression loss for driving commands
        'maintenance_output': 'binary_crossentropy'  # Classification loss for maintenance prediction
    },
    metrics={
        'driving_output': 'mae',
        'maintenance_output': 'accuracy'
    }
)

# Print model summary to verify architecture
model.summary()
