from pydub import AudioSegment

def m4a_to_wav(audio_file_path):
    sound = AudioSegment.from_file(audio_file_path, "mp4")
    audio_file_path = audio_file_path.replace(".m4a", '.wav')
    sound.export(audio_file_path, format="wav")
    return audio_file_path

audio_file_path = m4a_to_wav('n.m4a')

from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path

wav_fpath = Path(audio_file_path)

wav = preprocess_wav(wav_fpath)
print(2)
encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
print(cont_embeds.shape)

from spectralcluster import SpectralClusterer

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=100,
    p_percentile=0.90,
    gaussian_blur_sigma=1)

labels = clusterer.predict(cont_embeds)

def create_labelling(labels,wav_splits):
    from resemblyzer import sampling_rate
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            temp = [str(labels[i-1]),start_time,time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]),start_time,time]
            labelling.append(tuple(temp))

    return labelling
  
labelling = create_labelling(labels,wav_splits)
for i in labelling:
    print(i)
