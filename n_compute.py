from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from scipy.io import wavfile


wav_fpath = Path(".", "n.m4a")
wav = preprocess_wav(wav_fpath)
# wavfile.write('x.wav', sampling_rate, wav)

sec = lambda s: int(s * sampling_rate)

# Cut some segments from single speakers as reference audio
segments = [[104,107], [376,380]]
speaker_names = ["Natt"]
speaker_wavs = [ np.concatenate([wav[sec(s[0]):sec(s[1])] for s in segments]) ]

encoder = VoiceEncoder("cpu")
print("Running the continuous embedding on cpu, this might take a while...")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

# Get the continuous similarity for every speaker. It amounts to a dot product between the 
# embedding of the speaker and the continuous embedding of the interview
speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                   zip(speaker_names, speaker_embeds)}




import numpy
numpy.save("similarity_dict", similarity_dict)
numpy.save("wav_splits", wav_splits)


