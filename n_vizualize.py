from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from scipy.io import wavfile


## Get reference audios
# Load the interview audio from disk
# Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I
wav_fpath = Path(".", "n.m4a")
wav = preprocess_wav(wav_fpath)
# wavfile.write('x.wav', sampling_rate, wav)

sec = lambda s: int(s * sampling_rate)

# Cut some segments from single speakers as reference audio
segments = [[104,107], [376,380]]
speaker_names = ["Natt"]
speaker_wavs = [ np.concatenate([wav[sec(s[0]):sec(s[1])] for s in segments]) ]

# for w in speaker_wavs:
#     play_wav(w)




import numpy
similarity_dict = numpy.load("similarity_dict.npy",allow_pickle=True)
wav_splits = numpy.load("wav_splits.npy",allow_pickle=True)

# Run the interactive demo
interactive_diarization(similarity_dict, wav, wav_splits)


