import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
"Music Generator"
"----------------------------------------------------------------"
# import torch
# from transformers import AutoProcessor, MusicgenForConditionalGeneration
# 
# print(f"using device: ", device)

# processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
# model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
# inputs = processor(
#     text=["electric guitar rock solo"],
#     padding=True,
#     return_tensors="pt",
# ).to(device)

# model.to(device)
# audio_values = model.generate(**inputs, max_new_tokens=256)
# import scipy

# sampling_rate = model.config.audio_encoder.sampling_rate

# scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

"Audio Generator"
"----------------------------------------------------------------"

# import torchaudio
# from audiocraft.models import AudioGen
# from audiocraft.data.audio import audio_write

# model = AudioGen.get_pretrained('facebook/audiogen-medium')
# model.set_generation_params(duration=20)  # generate 8 seconds.
# # wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
# def iner_aud(inp):
#     descriptions = [inp]

#     wav = model.generate(descriptions)  # generates 3 samples.

#     for idx, one_wav in enumerate(wav):
#         # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#         audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
# iner_aud(input("enter text:"))

"Video Generator"
"----------------------------------------------------------------"

# import torch
# from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
# from diffusers.utils import export_to_video

# pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float32)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

# prompt = "a panda playing on a swing set"
# video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
# video_path = export_to_video(video_frames, output_video_path="./vid.mp4")

"Test to Image"
"----------------------------------------------------------------"
"1-5"
# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")

"2-1"
# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# model_id = "stabilityai/stable-diffusion-2-1"

# # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")

# prompt = "a couple walking in the woods with their dog"
# image = pipe(prompt).images[0]
    
# image.save("astronaut_rides_horse.png")

"Image to text"
"----------------------------------------------------------------"
# import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# # conditional image captioning
# text = "describe the photo"
# inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
# inputs = processor(raw_image, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))



# pip install accelerate
# import requests
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import torch
# from accelerate import infer_auto_device_map, init_empty_weights

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="cuda:0")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# question = "how many dogs are in the picture?"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
