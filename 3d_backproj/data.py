from   tensorpack import DataFlow #,JoinData,BatchData,logger
import numpy as np
import cfg

class ProjDataFlow(DataFlow):
    ''' produces movie sequence with batches of length framebatch and total length seqlen starting from a random location  '''
    def __init__(self,Pin):
        ''' reset_prob - probability that a sequence will get reset sometime before the end  '''
        # pins = [Pin]
        # for k in range(cfg.NCAND-1):
        #     idxs = np.arange(Pin.shape[0])
        #     np.random.shuffle(idxs)
        #     pins.append(Pin[idxs])

        self._Pin = Pin #np.stack(pins, axis=3)

    def get_data(self):
        while True:
            yield self._Pin[None,...]

    # def size(self):
    #     return int(np.ceil(self._seqlen/self._framebatch))

# def get_train_dataflow(Pin):
#     # generate batch movies dataflows with random start locations
#     df = ProjDataFlow(Pin)
#     return mdf