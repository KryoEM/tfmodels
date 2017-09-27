#### Global PARAMS #######
# picking diameter in pixels
PART_D_PIXELS = 64
# picking window size
PICK_WIN      = 12*PART_D_PIXELS

# particle picking strides
STRIDES       = [8]

# parameter for smoothing L1 bounding box loss
# SmoothL1(x) = 0.5 * (sigma * x) ^ 2,
# if | x | < 1 / sigma ^ 2
# | x | - 0.5 / sigma ^ 2, otherwise
L1_SIGMA = 5.0

# number of channels used for classification
CLS_CHANNELS   = 64
# number of channels in the first convolution layer
CONV0_CHANNELS = 16
# number of ausxilliary channels for focused classification
N_CLS_AUX_CHANNELS = 4
# focus loss function parameter
# GAMMA         = 0.0

# probability threshold for particle detection
PROB_THRESH  = 0.98
# how many locations in the perimeter match single particle hypothesis
PERIM_THRESH = 0.5

