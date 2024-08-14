import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

# Set up Streamlit app
st.title("Text-to-Speech with Custom Speaker Embedding")
st.write("This app generates speech from text using a custom speaker embedding.")

# Load the text-to-speech pipeline
@st.cache_resource
def load_tts_pipeline():
    return pipeline("text-to-speech", "microsoft/speecht5_tts")

synthesiser = load_tts_pipeline()

# Load the embeddings dataset
@st.cache_resource
def load_embeddings_dataset():
    return load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

embeddings_dataset = load_embeddings_dataset()

# Select a speaker embedding
selected_index = st.slider("Select a Speaker Embedding Index", 0, len(embeddings_dataset) - 1, 7306)
speaker_embedding = torch.tensor(embeddings_dataset[selected_index]["xvector"]).unsqueeze(0)

# Input text for speech synthesis
input_text = st.text_area("Enter the text you want to synthesize", "Welcome Guys.")

if st.button("Generate Speech"):
    with st.spinner("Generating speech..."):
        # Generate speech with the selected speaker embedding
        speech = synthesiser(input_text, forward_params={"speaker_embeddings": speaker_embedding})

        # Save the speech to a file
        output_file = "output_speech.wav"
        sf.write(output_file, speech["audio"], samplerate=speech["sampling_rate"])

        # Provide a download link for the generated speech
        st.audio(output_file)
        st.success(f"Speech generated successfully! You can download it from the link below.")
        st.download_button("Download Speech", data=open(output_file, "rb"), file_name=output_file)
