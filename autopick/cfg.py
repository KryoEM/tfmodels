#### Global PARAMS #######

WEIGHT_DECAY = 1e-7
ALPHA_DECAY  = 1e-5

DD = 8

# picking diameter in pixels
PART_D_PIXELS = 64
# picking window size
PICK_WIN      = 10*PART_D_PIXELS
# low pass resolution for background removal
LP_RES        = 4.0*PART_D_PIXELS

# particle picking strides
STRIDES       = [8]

# parameter for smoothing L1 bounding box loss
# SmoothL1(x) = 0.5 * (sigma * x) ^ 2,
# if | x | < 1 / sigma ^ 2
# | x | - 0.5 / sigma ^ 2, otherwise
L1_SIGMA = 5.0

# number of channels in the first convolution layer
CONV0_CHANNELS = 16
# number of auxilliary channels for focused classification
N_CLS_AUX_CHANNELS = 4
# number of channels used for focused classification
CLS_CHANNELS = 64

# probability threshold for particle detection
PROB_THRESH  = 0.9
# the minimum value for the particle pixel "clearance" from neighbor particles presence
# values in [0,1] 1-cleanest
MIN_CLEARANCE = 0.2

# whether to resize all micros, or use previously resized ones
RESIZE_MICROS = False


