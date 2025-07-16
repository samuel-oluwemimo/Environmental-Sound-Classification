import streamlit as st
import io
import requests
import base64
import soundfile as sf
import json
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Replace this with your actual deployed Modal endpoint URL
# You can get this from your Modal deployment logs or dashboard.
MODAL_API_URL = " https://samuel-oluwemimo--audio-cnn-inference-audioclassifier-inference.modal.run/"

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Enviromental Sound Classifier",
    page_icon="🎵",
    layout="wide"  # Changed layout to wide for more space
)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stAudio {
        border-radius: 8px;
        overflow: hidden;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎵 Environmental Sound Classification App")
st.info("Upload a WAV file and get its classification predictions and visualizations using a deployed model.")


# --- Helper function for plotting feature maps/spectrograms ---
def plot_2d_array_as_image(data_values, title, cmap='magma', aspect='auto', colorbar_label=''):
    """Plots a 2D array as an image using matplotlib and displays it in Streamlit."""
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figsize as needed
    im = ax.imshow(np.array(data_values), cmap=cmap, aspect= aspect, origin='lower')
    ax.set_title(title, fontsize=14)
    ax.axis('off')  # Hide axes for cleaner look

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, rotation=270, labelpad=15)

    st.pyplot(fig)
    plt.close(fig)  # Close the figure to free memory


# --- Main Content Area ---
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

col1, col2 = st.columns([1,4], gap="medium")

if uploaded_file is not None:
    with col1:
        # Get file details for validation
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.subheader("Uploaded File Details")
        st.write(f"**Filename**: {file_details['filename']}")
        st.write(f"**File Type**: {file_details['filetype']}")
        st.write(f"**File Size**: {file_details['filesize'] / (1024 * 1024):.2f} MB")

        # Define the maximum allowed file size in bytes (10 MB)
        max_file_size_mb = 10
        max_file_size_bytes = max_file_size_mb * 1024 * 1024

    with col2:
        # Check file size
        if uploaded_file.size > max_file_size_bytes:
            st.error(f"Error: File size exceeds the maximum limit of {max_file_size_mb} MB.")
            st.warning("Please upload a smaller WAV file.")
            uploaded_file = None  # Clear the file to prevent further processing
        else:
            st.success("File uploaded successfully! Processing...")

            # Read audio data from the uploaded file
            try:
                # Use uploaded_file.read() to get bytes, then BytesIO for soundfile
                audio_bytes_raw = uploaded_file.read()
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes_raw))

                # Encode audio to base64 for the API request
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sample_rate, format="WAV")
                audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                payload = {"audio_data": audio_b64}

                st.subheader("Sending to Model...")
                with st.spinner("Classifying audio and generating visualizations... This might take a moment."):
                    try:
                        # Make the POST request to the Modal endpoint
                        response = requests.post(MODAL_API_URL, json=payload, timeout=610)  # Client-side timeout
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        result = response.json()

                        st.subheader("Classification Results")

                        # Display Predictions
                        predictions = result.get("predictions", [])
                        if predictions:
                            st.write("#### Top Predictions:")
                            for pred in predictions:
                                st.markdown(
                                    f"- **{pred['class'].replace('_', ' ').title()}**: **`{pred['confidence']:.2%}`**")
                        else:
                            st.warning("No predictions received.")

                        # --- Visualizations ---
                        st.markdown("---")
                        st.header("Visualizations")

                        # Audio Playback
                        st.subheader("Audio Playback")
                        # Reset file pointer to read again for playback
                        uploaded_file.seek(0)
                        st.audio(uploaded_file.read(), format='audio/wav')

                        # Waveform Visualization
                        st.subheader("Audio Waveform")
                        waveform_info = result.get("waveform", {})
                        if waveform_info and waveform_info.get("values"):
                            st.line_chart(waveform_info["values"], use_container_width=True)
                            st.write(f"**Duration**: {waveform_info.get('duration', 0):.2f} seconds")
                            st.write(f"**Sample Rate**: {waveform_info.get('sample_rate', 0)} Hz")
                        else:
                            st.info("Waveform data not available.")

                        # Input Spectrogram Visualization
                        st.subheader("Input Spectrogram")
                        input_spectrogram = result.get("input_spectrogram", {})
                        if input_spectrogram and input_spectrogram.get("values"):
                            plot_2d_array_as_image(
                                input_spectrogram["values"],
                                f"Input Spectrogram (Shape: {input_spectrogram['shape'][0]}x{input_spectrogram['shape'][1]})",
                                cmap='magma',
                                colorbar_label='Amplitude (dB)'
                            )
                        else:
                            st.info("Input spectrogram data not available.")

                        # Feature Maps Visualization
                        st.subheader("Convolutional Layer Outputs")
                        visualization_data = result.get("visualization", {})
                        if visualization_data:
                            for layer_name, layer_data in visualization_data.items():
                                with st.expander(
                                        f"View {layer_name} (Shape: {layer_data['shape'][0]}x{layer_data['shape'][1]})"):
                                    plot_2d_array_as_image(
                                        layer_data["values"],
                                        f"{layer_name} Feature Map",
                                        cmap='viridis',  # A general colormap for feature maps
                                        colorbar_label='Activation Value'
                                    )
                        else:
                            st.info("Feature map visualization data not available.")

                    except requests.exceptions.Timeout:
                        st.error(
                            "The request to the classification model timed out. Please try again or with a smaller file.")
                    except requests.exceptions.ConnectionError:
                        st.error(
                            "Could not connect to the classification model. Please check your internet connection or the model's deployment status.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"An error occurred during the API request: {e}")
                    except json.JSONDecodeError:
                        st.error(
                            "Received an invalid response from the model. It might be an internal server error or malformed JSON.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during classification: {e}")

            except Exception as e:
                st.error(f"Error reading or processing the uploaded audio file: {e}")

