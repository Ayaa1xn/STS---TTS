# STT-TTS Pipeline with Jais LLM-Integration
This project demonstrates the integration of Speech-to-Text (STT) and Text-to-Speech (TTS) with the **Jais 13B Language Model** to process and generate audio/text data. The project utilizes **Coqui TTS** for TTS, **Whisper** for STT, and **Jais LLM** for text generation. It is ideal for educational purposes and small-scale experiments.
## Project Overview

1. **Speech-to-Text (STT)**:
   - Converts audio input (e.g., `.flac` or `.wav` files) into text using OpenAI's Whisper model.
2. **Text Processing with Jais LLM**:
   - Processes transcribed text using the **Jais 13B Large Language Model**, generating responses or performing natural language tasks.
3. **Text-to-Speech (TTS)**:
   - Converts the Jais-generated response back into speech using **Coqui TTS**.
4. **Audio Dataset**:
   - Uses **LibriSpeech UserLibri** dataset for testing. The dataset includes `.flac` files and associated metadata.
## **Folder Structure**
project-folder/
├── data/
│   ├── input_audio/         # Contains raw input audio files
│   ├── output_audio/        # Contains generated speech files from TTS
│   └── transcripts/         # Contains transcribed text from STT
├── notebooks/
│   ├── stt_pipeline.ipynb   # Whisper-based STT implementation
│   ├── tts_pipeline.ipynb   # Coqui TTS implementation
│   ├── jais_pipeline.ipynb  # Jais LLM text processing
├── offload_weights/         # Stores offloaded model weights for Jais LLM
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
## **Getting Started**

### **Prerequisites**
1. **Python Environment**: Ensure Python 3.9 or above is installed.
2. **GPU Support**: A CUDA-compatible GPU is recommended for faster processing.
3. **Install Dependencies**:
   Install all required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
### **Key Libraries**
- **STT**: OpenAI Whisper
- **TTS**: Coqui TTS
- **LLM**: Transformers for Jais 13B
## **Setup**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Dataset**
Download the **LibriSpeech UserLibri** dataset:
1. Install Kaggle:
   ```bash
   pip install kaggle
   ```
2. Authenticate Kaggle:
   - Place your `kaggle.json` API key in `~/.kaggle/kaggle.json`.
3. Download the dataset:
   ```bash
   kaggle datasets download -d google/userlibri --unzip -p data/
   ```
   Ensure that `.flac` audio files are in the `data/input_audio/` directory.

---
## **How to Run**

### **Step 1: Speech-to-Text (STT)**
Run the Whisper STT model to convert audio to text:
```python
from whisper import load_model

# Load Whisper model
model = load_model("base")

# Process audio file
audio_path = "data/input_audio/sample.flac"
result = model.transcribe(audio_path)
print("Transcription:", result["text"])

# Save the transcription
with open("data/transcripts/sample.txt", "w") as f:
    f.write(result["text"])
```

### **Step 2: Text Processing with Jais LLM**
Run the Jais model to process the transcribed text and generate a response:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Jais model and tokenizer
model_path = "inceptionai/jais-13b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    offload_folder="offload_weights"
)

# Process text with Jais
prompt = "Transcribed text: " + result["text"]
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=200)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Response from Jais:", response)
```

### **Step 3: Text-to-Speech (TTS)**
Run Coqui TTS to convert the Jais response into speech:
```python
from TTS.api import TTS

# Load Coqui TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# Generate speech
output_audio_path = "data/output_audio/response.wav"
tts.tts_to_file(text=response, file_path=output_audio_path)
print(f"Speech saved to {output_audio_path}")
```
## **What Does Jais Do in This Pipeline?**
Jais 13B is used as the language model to:
1. Process the transcribed text.
2. Generate responses or perform text-based tasks.
3. Enhance conversational abilities or analyze the input audio contextually.

For instance:
- Input transcription: `"How is the weather today?"`
- Jais response: `"The weather today is sunny with a mild breeze."`
- Final TTS output: The Jais-generated text is converted back to speech.

---

## **Troubleshooting**

1. **403 Forbidden Kaggle Error**:
   - Ensure `kaggle.json` is correctly placed in `~/.kaggle/`.
   - Verify API key permissions on Kaggle.

2. **Memory Issues**:
   - Use model offloading to prevent crashes when loading Jais 13B:
     ```python
     model = AutoModelForCausalLM.from_pretrained(
         model_path,
         device_map="auto",
         offload_folder="offload_weights"
     )
     ```

3. **File Not Found**:
   - Check if `.flac` files are correctly placed in the `data/input_audio/` folder.

---

## **Future Improvements**

- Add support for multilingual STT and TTS.
- Train Jais on custom datasets to fine-tune its performance.
- Implement real-time audio input/output for conversational AI systems.

---

## **Contributing**

Contributions are welcome! Feel free to submit issues or pull requests.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
