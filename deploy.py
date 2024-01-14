import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import librosa
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def loadmodel():

  # Load the model
  model = load_model('model.h5')
  return model

model = loadmodel()

st.header("""
          Sound Event Localisation and Detection Using Machine Learning
          """)

file = st.file_uploader("Kindly upload the audio file.",type = ["wav"])


if file:
    wav, sr = librosa.load(file, sr=None)

# Extract MFCC features
    mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)

# Extract intensity vector (RMS energy)
    intensity = librosa.feature.rms(y=wav)

# Concatenate features along the first axis
    features = np.concatenate((mfccs, intensity), axis=0)

# Transpose to have channels as the last dimension
    features = np.transpose(features)

# Check the number of channels and pad if necessary
    num_channels_expected = 7
    if features.shape[-1] < num_channels_expected:
        features = np.pad(features, ((0, 0), (0, num_channels_expected - features.shape[-1])))

# Resample the features to the target shape using linear interpolation
    target_shape = (128, 64)
    resampled_features = interp2d(
        np.linspace(0, 1, features.shape[1]),
        np.linspace(0, 1, features.shape[0]),
        features,
        kind='linear',
    )(np.linspace(0, 1, target_shape[1]), np.linspace(0, 1, target_shape[0]))

# Add batch dimension and reshape to match the expected input shape
    input_data = np.expand_dims(resampled_features, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

# Repeat the input along the channel dimension to match the expected channels
    input_data = np.repeat(input_data, num_channels_expected, axis=-1)

# Make predictions
    predictions_list = model.predict(input_data)

    class SoundEventPostprocessor:
        def __init__(self):
            self._unique_classes = {
                'clearthroat': 2,
                'cough': 8,
                'doorslam': 9,
                'drawer': 1,
                'keyboard': 6,
                'keysDrop': 4,
                'knock': 0,
                'laughter': 10,
                'pageturn': 7,
                'phone': 3,
                'speech': 5
            }

            self.tracked_sound_events = {}

        def postprocess(self, model_output, timestamp):
            # Assuming model_output is a NumPy array
            if isinstance(model_output, np.ndarray):
                # Assuming that azimuth, elevation, distance are present at specific indices
                azimuths = model_output[..., 0]  # Adjust the index according to your model's output
                elevations = model_output[..., 1]
                distances = model_output[..., 2]
                class_probs = model_output[..., 3:]  # Assuming the rest are class probabilities

                # Thresholding - Example threshold value (adjust as needed)
                threshold = 0.5
                detection_mask = class_probs.max(axis=-1) > threshold
                azimuths = azimuths[detection_mask]
                elevations = elevations[detection_mask]
                distances = distances[detection_mask]
                class_ids = class_probs.argmax(axis=-1)[detection_mask]

                # Tracking sound events over time
                self._track_sound_events(azimuths, elevations, distances, class_ids, timestamp)

                # Other postprocessing steps as needed...

        def _track_sound_events(self, azimuths, elevations, distances, class_ids, timestamp):
            # Dummy tracking logic, replace with your tracking algorithm
            for i in range(len(azimuths)):
                sound_event_id = f"{class_ids[i]}_{i}"  # Unique ID for each sound event
                if sound_event_id in self.tracked_sound_events:
                    # Update existing tracked sound event
                    self.tracked_sound_events[sound_event_id]['azimuth'].append(azimuths[i])
                    self.tracked_sound_events[sound_event_id]['elevation'].append(elevations[i])
                    self.tracked_sound_events[sound_event_id]['distance'].append(distances[i])
                    self.tracked_sound_events[sound_event_id]['timestamps'].append(timestamp)
                else:
                    # Create a new tracked sound event
                    self.tracked_sound_events[sound_event_id] = {
                        'azimuth': [azimuths[i]],
                        'elevation': [elevations[i]],
                        'distance': [distances[i]],
                        'timestamps': [timestamp],
                        'class_id': class_ids[i]
                    }

            # Remove inactive sound events (e.g., not detected in recent frames)
            self._remove_inactive_sound_events()

        def _remove_inactive_sound_events(self):
            # Dummy logic to remove inactive sound events (replace with your logic)
            for sound_event_id in list(self.tracked_sound_events.keys()):
                # Check if the sound event is inactive based on your criteria
                if len(self.tracked_sound_events[sound_event_id]['azimuth']) > 5:
                    del self.tracked_sound_events[sound_event_id]

        def save_tracked_events_to_file(self, file_path='tracked_events.txt'):
            with open(file_path, 'w') as file:
                file.write("Tracked Sound Events:\n")
                for sound_event_id, details in self.tracked_sound_events.items():
                    file.write(f"Sound Event ID: {sound_event_id}\n")
                    file.write(f"Azimuths: {details['azimuth']}\n")
                    file.write(f"Elevations: {details['elevation']}\n")
                    file.write(f"Distances: {details['distance']}\n")
                    file.write(f"Timestamps: {details['timestamps']}\n")
                    file.write(f"Class ID: {details['class_id']}\n")
                    file.write("------\n")


    postprocessor = SoundEventPostprocessor()
    # Assuming predictions_list is a list of dictionaries
    timestamp = 0.0  # Initial timestamp
    for predictions in predictions_list:
        # Call the postprocess method with each set of predictions
        postprocessor.postprocess(predictions, timestamp)
        timestamp += 1.0  # Update timestamp for the next frame

    # Print or save the predictions
    print(predictions_list)
    # Assuming the predictions are stored in the variable 'predictions'
    probabilities_array, logits_array = predictions_list

    # Apply softmax to get probabilities
    class_probabilities = np.exp(probabilities_array) / np.sum(np.exp(probabilities_array), axis=-1, keepdims=True)

    # Get predicted class indices
    predicted_classes = np.argmax(class_probabilities, axis=-1)

    class SoundEventVisualizer:
        def __init__(self, sound_event_postprocessor):
            self.postprocessor = sound_event_postprocessor

        def plot_3d(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'H', 'D', 'x']
            class_labels = {v: k for k, v in self.postprocessor._unique_classes.items()}

            for class_id, marker in zip(range(len(self.postprocessor._unique_classes)), markers):
                for sound_event_id, details in self.postprocessor.tracked_sound_events.items():
                    if details['class_id'] == class_id:
                        azimuths = details['azimuth']
                        elevations = details['elevation']
                        distances = details['distance']
                        timestamps = details['timestamps']

                        # Plotting with respect to time
                        ax.scatter(timestamps, azimuths, elevations, marker=marker, label=f"Class {class_labels[class_id]}")

            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Azimuth')
            ax.set_zlabel('Elevation')
            ax.legend()

            default_save_path = os.path.join(os.getcwd(), 'output_plot.png')
            if os.path.exists(default_save_path):
                os.remove(default_save_path)
            plt.savefig(default_save_path, format='png')
            plt.close()


    # Assuming predictions_list is a list of dictionaries
    timestamp = 0.0  # Reset timestamp for visualization
    for predictions in predictions_list:
        # Call the postprocess method with each set of predictions
        postprocessor.postprocess(predictions, timestamp)
        timestamp += 1.0  # Update timestamp for the next frame

    # Visualize tracked events
    visualizer = SoundEventVisualizer(postprocessor)

    visualizer.plot_3d()
    postprocessor.save_tracked_events_to_file()
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    tracked = '/Users/subramanyam/Desktop/streamlit/tracked_events.txt'


    st.write("The Predicted Outputs are:")
    st.subheader("The Class Probabilities are:")
    st.success(class_probabilities)
    st.subheader("The Predicted classes are:")
    st.success(predicted_classes)
    if tracked is not None:
            # Read and display the content if a file is uploaded
            file_contents = read_text_file(tracked)
            st.subheader("The Tracked Sound Events are:")
            st.text(file_contents)
    st.subheader("The Visualised output is:")
    st.image('/Users/subramanyam/Desktop/streamlit/output_plot.png')

