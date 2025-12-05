import sys
import pickle

# Make sure Python can find the src/ package
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer_diphone import trainModel

# ----- experiment name -----
modelName = 'speechBaseline4_diphone'  # new name for diphone experiment

args = {}

# ----- paths -----
args['outputDir'] = '/home/jupyter/baseline_logs/' + modelName

# IMPORTANT: use the *diphone* dataset you created
args['datasetPath'] = '/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc_diphone'

# ----- load dataset once to set nClasses from diphone mapping -----
with open(args['datasetPath'], "rb") as f:
    data = pickle.load(f)

# This was created in make_diphone_dataset.py
n_diphones = len(data["diphone_to_id"])
print("Number of diphone classes (excluding blank):", n_diphones)

args['nClasses'] = n_diphones  # instead of 40 phonemes

# ----- model / training hyperparameters -----
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128

# If your trainer is using Adam (original baseline), keep these:
args['lrStart'] = 0.05
args['lrEnd']   = 0.02

# If your trainer is using AdamW (with eps=1e-8), use smaller LRs:
# args['lrStart'] = 0.001     # AdamW start LR
# args['lrEnd']   = 0.0001    # AdamW end LR

args['nUnits'] = 256
args['nBatch'] = 10000  # 3000 for quick tests, 10000 for full training
args['nLayers'] = 5
args['seed'] = 0
args['nInputFeatures'] = 256
args['dropout'] = 0.2

# augmentations
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0

# temporal parameters
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False

# regularization
args['l2_decay'] = 1e-5      # for Adam baseline;
# for AdamW you can also experiment with slightly larger, e.g. 1e-4 or 1e-3 later

# ----- train -----
trainModel(args)
