import whisperx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
#
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
# Set the device to CPU if CUDA is not available
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4  # Adjust based on your local machine's GPU memory
compute_type = "int8"  # Use "int8" for lower memory usage (may affect accuracy)
# Specify the path to your audio file
audio_file = "/home/knoldus/CouncilMeetingSample.mp3"  # Update the path to the audio file
# Load the audio file
audio = whisperx.load_audio(audio_file)
# Load the model
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
# Transcribe the audio
result = model.transcribe(audio, batch_size=batch_size)
# print(result)
result_text = ''.join(segment['text'] for segment in result['segments'])
# print(result["segments"])  # Print the segments before alignment
print(result_text)
# Align the whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
# print(result)
