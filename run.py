"Music Generator"
"----------------------------------------------------------------"
# from transformers import AutoProcessor, MusicgenForConditionalGeneration


# processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
# model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# inputs = processor(
#     text=["snow and wind in the wild", "a dog, bird and a fox"],
#     padding=True,
#     return_tensors="pt",
# )

# audio_values = model.generate(**inputs, max_new_tokens=256)

# import scipy

# sampling_rate = model.config.audio_encoder.sampling_rate
# scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

"Audio Generator"
"----------------------------------------------------------------"

# import torchaudio
# from audiocraft.models import AudioGen
# from audiocraft.data.audio import audio_write

# model = AudioGen.get_pretrained('facebook/audiogen-medium')
# model.set_generation_params(duration=20)  # generate 8 seconds.
# # wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
# descriptions = ['a jungle noises birds monkeys and insects with a big waterfall in the distance']
# wav = model.generate(descriptions)  # generates 3 samples.

# for idx, one_wav in enumerate(wav):
#     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#     audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)


"Video Generator"
"----------------------------------------------------------------"

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "a couple walking with their dog in the forest"
video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=48).frames
video_path = export_to_video(video_frames, output_video_path="./vid.mp4")
