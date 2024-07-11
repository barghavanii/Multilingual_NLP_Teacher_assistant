# -*- coding: utf-8 -*-

# -- Sheet --

pip install pypdf
pip install langchain
pip install chromadb
pip install openai
pip install tiktoken
pip install chroma-migrate
sudo apt update && sudo apt install ffmpeg
pip install git+https://github.com/openai/whisper
pip install pydub

!pwd

%cd TTS

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ChatVectorDBChain
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import os

# Replace YOUR_OPENAI_API_KEY with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = "%API KEY"

pdf_path= '/data/notebook_files/2301.02111.pdf'
loader= PyPDFLoader(pdf_path)
pages= loader.load_and_split()
print(pages[0].page_content)  # Access the attribute directly without calling it as a method

embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()

import whisper
model = whisper.load_model("large")

from pydub import AudioSegment

# Load .ogg file
audio = AudioSegment.from_ogg("/data/notebook_files/WhatsApp Ptt 2024-04-09 at 10.04.12 PM.ogg")

# Export to .mp3
audio.export("/data/notebook_files/query.mp3", format="mp3")

result = model.transcribe("/data/notebook_files/query.mp3", language="en",fp16=False)
print(result["text"])

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Assuming vectordb is already initialized correctly
pdf_ga = ChatVectorDBChain.from_llm(
    ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    vectordb, return_source_documents=True
)
query = result["text"]
result=pdf_ga({"question":query,"chat_history":""})
print("Answare:")
print(result["answer"])

from txtai.pipeline import Translation

# Create translation model
translate = Translation()
translation = translate(result["answer"], "fa")
translation

translation

def split_text_into_chunks(text, chunk_size=400):
    """
    Splits a text into chunks of a specified size.

    Args:
    - text (str): The text to be split.
    - chunk_size (int): The maximum size of each chunk in characters.

    Returns:
    - list: A list of text chunks.
    """
    # Split the text into chunks of 'chunk_size' characters
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Your long text goes here
long_text =translation

# Splitting the text
chunks = split_text_into_chunks(long_text, 400)

# You can then iterate over each chunk to process it
for chunk in chunks:
    print(chunk)
    # Here you can call your TTS model to process each chunk

# Azure demo
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf  # Ensure this library is installed
import os  # For creating directories and path manipulation

# Function to split the text into chunks
def split_text_into_chunks(text, chunk_size=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Load the configuration
config_path = "//data/notebook_files/XTTS_Azure/config (4).json"
config = XttsConfig()
config.load_json(config_path)

# Initialize the model from the configuration
model = Xtts.init_from_config(config)

# Load the model checkpoint
checkpoint_dir = "/data/notebook_files/XTTS_Azure"
model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True)

# Move the model to GPU if you're using CUDA
model.cuda()

# Your long translation text
long_translation = translation

# Splitting the translation text into chunks
chunks = split_text_into_chunks(long_translation, 100)

# Define the base directory where you want to save the output audio files
results_dir = "/data/notebook_files/results"
os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Iterate over each chunk and synthesize speech
for i, chunk in enumerate(chunks):
    # Synthesize speech for the current chunk
    outputs = model.synthesize(
        chunk,
        config,
        speaker_wav="/data/notebook_files/10-101-fa (1).wav",
        gpt_cond_len=3,
        language="fa",
    )

    # Use 'outputs['wav']' to get the audio data since it's stored under the 'wav' key
    audio_data = outputs['wav']
    samplerate = 22050  # Adjust this based on your model's output sample rate

    # Specify where you want to save the output audio file for the current chunk
    output_path = os.path.join(results_dir, f"synthesized_speech_chunk_{i+1}.wav")

    # Save the audio data
    sf.write(file=output_path, data=audio_data, samplerate=samplerate)

    print(f"Audio saved to {output_path}")

chunks

from IPython.display import Audio

# Use 'outputs['wav']' to get the audio data since it's stored under the 'wav' key
audio_data = "/data/notebook_files/results/synthesized_speech_chunk_5.wav"
samplerate = 22050  # Adjust this based on your model's output sample rate

# Instead of saving, directly play it in the notebook
Audio(audio_data, rate=samplerate)



