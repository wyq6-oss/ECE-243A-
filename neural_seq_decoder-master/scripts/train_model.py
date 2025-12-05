import sys
sys.path.append('/home/jupyter/neural_seq_decoder-master/src')

from neural_decoder.neural_decoder_trainer import trainModel

modelName = 'speechBaseline4_sample'
args = {}
args['outputDir'] = '/home/jupyter/baseline_logs/' + modelName
args['datasetPath'] = '/home/jupyter/neural_seq_decoder-master/competitionData/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
# args['lrStart'] = 0.001     # AdamW
# args['lrEnd']   = 0.0001    # AdamW

args['nUnits'] = 256
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5
# args['l2_decay'] = 1e-2 # Adamw
trainModel(args)
