##
import  numpy as np
from    fileio import mrc
from   myplotlib import imshow,clf
import  os
import  glob
import  png # needs py
from    fileio import filetools as ft
# from    utils import tprint
import  pyprind

#---------------- FUNCTIONS ----------------------
def float2uint16(im):
    ''' converts gain corrected and averaged counts to uint16 '''
    #im =im - im.min() #im/=im.max()
    return np.uint16(256*im) #np.uint16(256*256*im)

def mrcs2pngs(fnames,label_dir,split,nlim):
    '''Writes mrc files as png files'''
    ft.mkdir_assure(label_dir)
    nfiles = 0
    for f in fnames:
        nfiles += mrc.shape(f)[0]
    nfiles = int(min(nlim,nfiles))
    bar    = pyprind.ProgBar(nfiles,stream = 1,title='Writing %d %s pngs into %s ...' % (nfiles,split,label_dir))
    count  = 0
    for f in fnames:
        ims = mrc.load(f)
        for idx in range(ims.shape[0]):
            if count == nfiles: break;
            png.from_array(float2uint16(ims[idx]),'L').save(os.path.join(label_dir,ft.file_only(f)+'_%d.png' % idx))
            count += 1
            bar.update()

def pour_mrcs2pngs(data_in,data_out,label,test_ratio=0.1,use_only=np.inf):
    '''Coarseley split data into train and test parts according to test_ratio.'''
    train_dir = os.path.join(data_out, 'train/%s' % label)
    test_dir  = os.path.join(data_out, 'test/%s' % label)
    fnames    = glob.glob(os.path.join(data_in, '*.mrcs'))
    nfiles    = len(fnames)
    # shuffle files in place
    np.random.shuffle(fnames[:nfiles])
    ntrain = np.int((1.0 - test_ratio) * nfiles)
    ftrain = fnames[:ntrain]
    ftest  = fnames[ntrain:]
    #tprint('Splitting %d input files into --> [%d train, %d test]' % (nfiles, ntrain, nfiles - ntrain))
    # write train images
    mrcs2pngs(ftrain,train_dir,'train',nlim=use_only*(1.0-test_ratio))
    # write test images
    mrcs2pngs(ftest,test_dir,'test',nlim=use_only*test_ratio)
#--------------------------------------------



##