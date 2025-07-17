## Environmental Sound Classification
![Alt]( ./1.png)

App link: https://environmental-sound-classification-mpznmnctvxkzjgvfqdipyd.streamlit.app/

Note: file must be wav and it must no be more than 10mb

### Overview
In this project, i learnt to train and deploy an audio classification CNN from scratch using PyTorch. The model classifies sounds like a dog barking or birds chirping from an audio file. I worked with advanced techniques like Residual Networks (ResNet), data mixing, and Mel Spectrograms to build a robust training pipeline. Afterwards, a dashboard was built using Streamlit to upload audio and visualize the model's internal layers.

### Features
- Deep Audio CNN for sound classification
- ResNet-style architecture with residual blocks
- Mel Spectrogram audio-to-image conversion
- Data augmentation with Mixup & Time/Frequency Masking
- Serverless GPU inference with Modal
- Visualization of internal CNN feature maps
- Real-time audio classification with confidence scores
- Waveform and Spectrogram visualization
- FastAPI inference endpoint
- Optimized training with AdamW & OneCycleLR scheduler
- TensorBoard integration for training analysis
- Batch Normalization for stable & fast training
- Streamlit
- Pydantic data validation for robust API requests


### Clone the Repository
```bash
git clone https://github.com/samuel-oluwemimo/Environmental-Sound-Classification.git
```


### Backend
Navigate to folder
```bash
cd Environmental-Sound-Classification
```

Install dependencies
```bash
 pip install -r requirements.txt
 ```

Modal setup
```bash
modal setup
 ```

Run on modal
```bash
modal run main.py
 ```

Deploy backend
```bash
modal deploy main.py
 ```


### Streamlit frontend
```bash
streamlit run app.py
```

