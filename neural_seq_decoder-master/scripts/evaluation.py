import pickle
import numpy as np

output_dir = "/home/jupyter/baseline_logs/speechBaseline4_diphone"  # your folder

with open(output_dir + "/trainingStats", "rb") as f:
    stats = pickle.load(f)

testCER = stats["testCER"]  # CER every 100 batches

print("All CER values during training:", testCER)
print("Best CER:", testCER.min())
print("Mean CER across evaluations:", testCER.mean())
print("Std of CER across evaluations:", testCER.std())
