import sys
import os
chatterbox=f"{os.getcwd()}/chatterbox/src"
sys.path.append(chatterbox)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import tempfile
import random
import numpy as np
import torch

from sentencex import segment
import re
from tqdm.auto import tqdm
import os
import shutil
import soundfile as sf
import uuid
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random

temp_audio_dir="./cloned_voices"
os.makedirs(temp_audio_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    current_model = get_or_load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")
    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt

    wav = current_model.generate(
        text_input,
        language_id=language_id,
        **generate_kwargs
    )
    return current_model.sr, wav.squeeze(0).numpy()

supported_languages = {
    "English": "en",
    "Hindi": "hi",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Spanish": "es"
}

def clean_text(text):
    replacements = {"–": " ", "—": " ", "-": " ", "**": " ", "*": " ", "#": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tts_file_name(text, language="en"):
    global temp_audio_dir
    random_string = uuid.uuid4().hex[:8].upper()
    file_name = f"{temp_audio_dir}/kushi_gen_{random_string}.wav"
    return file_name

def remove_silence_function(file_path,minimum_silence=50):
    output_path = file_path.replace(".wav", "_clean.wav")
    sound = AudioSegment.from_file(file_path, format="wav")
    audio_chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-45, keep_silence=minimum_silence)
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format="wav")
    return output_path

def clone_voice_streaming(
    text,
    audio_prompt_path_input,
    lang_name="English",
    exaggeration_input=0.5,
    temperature_input=0.8,
    seed_num_input=0,
    cfgw_input=0.5,
    stereo=False,
    remove_silence=False,
    remove_noise=False,
):
    # Ensure default_ref.wav is used if nothing is uploaded
    if not audio_prompt_path_input or not os.path.exists(audio_prompt_path_input):
        audio_prompt_path_input = "default_ref.wav"

    language_id = supported_languages.get(lang_name, "en")
    text = clean_text(text)
    
    final_path = tts_file_name(text, language_id)
    samplerate = 24000
    channels = 2 if stereo else 1

    with sf.SoundFile(final_path, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_16') as f:
        sr, audio = generate_tts_audio(text, language_id, audio_prompt_path_input, exaggeration_input, temperature_input, seed_num_input, cfgw_input)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1) if stereo else audio[:, None]
        f.write(audio)
    
    if remove_silence:
        final_path = remove_silence_function(final_path)
    return final_path    

def tts_only(text, audio_prompt_path_input, lang_name="English", exaggeration_input=0.5, temperature_input=0.8, seed_num_input=0, cfgw_input=0.5, remove_silence=False, stereo=False):
    path = clone_voice_streaming(text, audio_prompt_path_input, lang_name, exaggeration_input, temperature_input, seed_num_input, cfgw_input, stereo, remove_silence)
    return path, path

# --- CUSTOMIZED CONFIG ---
LANGUAGE_CONFIG = {
    "en": {
        "audio": "default_ref.wav",
        "text": "Welcome to the Kushi-v1 Voice Hub. Enter text here to synthesize her voice."
    },
    "hi": {
        "audio": "default_ref.wav",
        "text": "कुशी-v1 वॉयस हब में आपका स्वागत है। उसकी आवाज़ बनाने के लिए यहाँ टेक्स्ट लिखें।"
    }
}

def default_audio_for_ui(lang_name):
    return "default_ref.wav"

def default_text_for_ui(lang_name):
    lang_code = supported_languages.get(lang_name, "en")
    return LANGUAGE_CONFIG.get(lang_code, {}).get("text", "Welcome to Kushi-v1.")

import gradio as gr
def tts_ui():
  custom_css = """.gradio-container { background-color: #0b0f19; color: white; }"""
  with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
      gr.HTML("""
        <div style="text-align: center; margin: 20px auto;">
            <h1 style="font-size: 2.8em; color: white;">🎙️ Kushi-v1 Voice Hub</h1>
            <p style="color: #9ca3af;">Personalized Neural Voice Synthesis System</p>
        </div>""")
      with gr.Row():
          with gr.Column():
              text = gr.Textbox(value=default_text_for_ui("English"), label="Text to synthesize", lines=4)
              language_id = gr.Dropdown(choices=list(supported_languages.keys()), value="English", label="Language Selection")
              ref_wav = gr.Audio(type="filepath", label="Voice Reference (Default: Kushi)", value="default_ref.wav")
              run_btn = gr.Button("Generate Voice", variant="primary")
              Remove_Silence_button = gr.Checkbox(label="Trim Silence", value=True)
          with gr.Column():
              audio_output = gr.Audio(label="Synthesized Output")
              audio_file = gr.File(label="Download WAV File")

      run_btn.click(
          fn=tts_only,
          inputs=[text, ref_wav, language_id, gr.State(0.5), gr.State(0.8), gr.State(0), gr.State(0.5), Remove_Silence_button, gr.State(False)],
          outputs=[audio_output, audio_file],
      )
  return demo

if __name__ == "__main__":
    demo = tts_ui()
    demo.queue().launch(share=True)