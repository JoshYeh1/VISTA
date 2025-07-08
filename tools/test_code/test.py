import numpy as np
import wave
from projectaria_tools.core import data_provider
import argparse

# -------------------- CLI --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--vrs", default="/Users/joshuayeh/dataset_project/VISTA/data/raw/20250618_objectloc_office.vrs")
args = parser.parse_args()

# ----------------- Open provider -------------
provider = data_provider.create_vrs_data_provider(args.vrs)
if provider is None:
    raise RuntimeError("Invalid VRS file")
stream_id = provider.get_stream_id_from_label("mic")
n_audio = provider.get_num_data(stream_id)
all_samples = []

output_wav = "output_multichannel.wav"
sample_rate = 48000
n_channels = 7  # change this if your setup uses a different count

provider = data_provider.create_vrs_data_provider(args.vrs)
stream_id = provider.get_stream_id_from_label("mic")
n_audio = provider.get_num_data(stream_id)

# Load audio blocks
samples = []
for i in range(n_audio):
    audio_data = provider.get_audio_data_by_index(stream_id, i)
    raw_block = audio_data[0].data
    samples.append(np.asarray(raw_block, dtype=np.int32))


block = provider.get_audio_data_by_index(stream_id, 0)[0].data
print(f"Block length: {len(block)}")
for n in range(1, 11):
    if len(block) % n == 0:
        print(f"{n} channels possible: block has {len(block) // n} frames")
block = provider.get_audio_data_by_index(stream_id, 0)[0].data
block_np = np.asarray(block)
print("First 28 values:", np.round(block_np[:28], 4))

# Flatten and reshape
flat_audio = np.concatenate(samples)
assert len(flat_audio) % n_channels == 0, "Audio length not divisible by number of channels"
multi_audio = flat_audio.reshape(-1, n_channels)

# Normalize to 16-bit PCM
max_val = np.max(np.abs(multi_audio))
pcm16 = ((multi_audio / max_val) * 32767).astype(np.int16)

# Save to .wav
with wave.open(output_wav, "wb") as wf:
    wf.setnchannels(n_channels)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sample_rate)
    wf.writeframes(pcm16.tobytes())

print(f"Wrote multichannel audio to: {output_wav} ({n_channels} ch, {sample_rate} Hz)")
