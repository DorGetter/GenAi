import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

# text to image.
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def text2image(pmt):
    image = pipe(pmt).images[0]
    image.save("astronaut_rides_horse.png")
    return image

# image to text

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor2 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model2 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
def image2text(img, pmt):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
   # unconditional image captioning
    inputs = processor2(raw_image, return_tensors="pt").to("cuda")

    out = model2.generate(**inputs)
    print(processor2.decode(out[0], skip_special_tokens=True))
"Audio Generator"
"----------------------------------------------------------------"



model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=20)  # generate 8 seconds.
# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
def text2sound(pmt):
    descriptions = [pmt]

    wav = model.generate(descriptions)  # generates 3 samples.

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    


def main():
    image = text2image(input("enter prompt: "))
    pmt = image2text(image, "what in the photo")
    text2sound(pmt)

# __name__
if __name__=="__main__":
    main()