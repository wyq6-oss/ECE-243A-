import pickle
import numpy as np

in_path = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc"
out_path = "/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc_diphone"

with open(in_path, "rb") as f:
    data = pickle.load(f)

diphone_to_id = {}
next_id = 1  # 0 is reserved for CTC blank


def seq_to_diphone_ids(phone_seq):
    """
    phone_seq: 1D array of phoneme IDs (ints), length L.
    Returns: 1D array of diphone IDs (ints), length L-1.
    """
    global next_id

    phone_seq = np.asarray(phone_seq, dtype=np.int32)
    # remove any padding zeros just in case
    phone_seq = phone_seq[phone_seq != 0]

    if len(phone_seq) < 2:
        # no diphones possible
        return np.zeros((0,), dtype=np.int32)

    diphone_ids = []
    for i in range(len(phone_seq) - 1):
        pair = (int(phone_seq[i]), int(phone_seq[i + 1]))
        if pair not in diphone_to_id:
            diphone_to_id[pair] = next_id
            next_id += 1
        diphone_ids.append(diphone_to_id[pair])

    return np.array(diphone_ids, dtype=np.int32)


def transform_split(split_data):
    """
    split_data: data['train'], data['test'], or data['competition'].
    Each entry is a 'day' dict with keys including 'phonemes' and 'phoneLens'.
    """
    for day in split_data:
        ph_list = day["phonemes"]     # list/array of phoneme sequences per trial
        lens_list = day["phoneLens"]  # lengths per trial

        new_ph_list = []
        new_len_list = []

        for i, ph_seq in enumerate(ph_list):
            L = int(lens_list[i])
            ph_seq = np.asarray(ph_seq, dtype=np.int32)[:L]  # trim to true length

            diphone_seq = seq_to_diphone_ids(ph_seq)
            new_ph_list.append(diphone_seq)
            new_len_list.append(len(diphone_seq))

        day["phonemes"] = new_ph_list
        day["phoneLens"] = np.array(new_len_list, dtype=np.int32)


# ---- apply to all splits we care about ----
transform_split(data["train"])
transform_split(data["test"])
if "competition" in data:
    transform_split(data["competition"])

print("Number of diphone classes (excluding blank):", len(diphone_to_id))

# Save mapping so you can decode later if needed
data["diphone_to_id"] = diphone_to_id
data["id_to_diphone"] = {v: k for k, v in diphone_to_id.items()}

with open(out_path, "wb") as f:
    pickle.dump(data, f)

print("Saved diphone dataset to:", out_path)
