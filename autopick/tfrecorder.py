import tensorflow as tf
import os
import dataset_utils
from   dataset_utils      import DataRaw,read_dataset_meta
from   tfrecorder_generic import Directory2TFRecord
from   fileio.filetools   import list_dirtree
from   scipy import misc
import numpy as np
from   fileio import filetools as ft
from   fileio import mrc
from   joblib import Parallel, delayed
# import multiprocessing as mp
import random
import cv2
from   myplotlib import imshow,clf,savefig
from   matplotlib import pyplot as plt
from   scipy.spatial.distance import cdist
from   autopick import cfg
import image
from   utils import tprint
import functools
import multiprocessing as mp
from   multiprocessing.dummy import Pool as ThreadPool
from   relion_params import path2psize,parse_particles_star,path2part_diameter

slim   = tf.contrib.slim

#### TFRECORDER PARAMS #######
LINE_WIDTH = 2.0
N_EXAMPLES = 500
IMAGE_KEY  = 'image/uint8'
SHAPE_KEY  = 'shape'
BACKGROUND = 'background'
CLEAN      = 'clean'
# flags
RESIZE_MICROS = True
AUGMENT_BACKGROUND = False
##########################

##
def ravel_coords_append(v,sz):
    res = np.int64(np.ravel_multi_index(v.transpose(), sz) + 1)
    return np.append(res,np.int64([0,0]))

def ravel_coords(v,sz):
    return np.int64(np.ravel_multi_index(v.transpose(), sz) + 1)

def unravel_coords_append(idxs,sz):
    return np.int32(np.unravel_index(idxs[:-2]-1, sz)).transpose()

##

def create_instance(*args,**kwargs):
    return ParticleCoords2TFRecord(*args,**kwargs)

def create_dataset(split_name, data_dir):
    dataset_meta = read_dataset_meta(split_name, data_dir)
    example_meta = dataset_meta['example_meta']
    coordlabels  = example_meta['classes']
    items_to_handlers = {
        'image':   DataRaw(label = IMAGE_KEY, shape_key=SHAPE_KEY, dtype=example_meta['dtype'])
    }
    # add handlers for each class coordinates
    for label in coordlabels:
        items_to_handlers.update({label:  DataRaw(label = label, shape=(-1,), dtype=tf.int64)})

    # add indices for coordinate conversions
    items_to_handlers.update({'dxyidxs': DataRaw(label='dxyidxs', shape=(-1,), dtype=tf.int64)})
    # add coordinate conversions
    items_to_handlers.update({'dxy': DataRaw(label='dxy', shape=(-1,2), dtype=tf.float32)})

    return dataset_utils.create_dataset(split_name,data_dir,items_to_handlers,dataset_meta)

def psize2bn(psize):
    return 2.0 * psize / cfg.RPN_RES

def calc_micro_pick_shape(micro,D_pixels):
    # read particle diameter
    # calculate binning that brings particle to canonic size
    bn = D_pixels / cfg.PART_D_PIXELS
    # calculate resized window size
    return np.int32(np.round(np.float32(mrc.shape(micro)[1:]) / bn)),bn

def plot_class_coords(im,class_coords,d):
    ax  = plt.subplot()
    plt.ion()
    vmn = np.percentile(im, 1)
    vmx = np.percentile(im, 99)
    ax.imshow(im, vmin=vmn, vmax=vmx, cmap=plt.cm.gray)
    colors  = ['b','r','g','c','m','y','k','w']
    idx     = 0
    cstr    = ''
    classes = ''
    for key in class_coords:
        for coord in class_coords[key]:
            y, x = coord
            if x >= 0:
                circ = plt.Circle((x, y), radius=d / 2.0, color=colors[idx], fill=False, lw=LINE_WIDTH)
                ax.add_patch(circ)
            # im[coord[0],coord[1]] = 255
        cstr += colors[idx] + ','
        classes += key + ','
        idx  += 1
        ax.axis((0, im.shape[1], im.shape[0], 0))
    ax.set_title("%s = %s" % (cstr[:-1],classes[:-1]))
    plt.draw()
    plt.pause(0.001)


def neib_coords(coords,D):
    ''' Turns each coordinate to a set of coordinates insize a circular mask '''
    R      = D//2
    marg   = np.arange(-R, R + 1)
    dx, dy = np.meshgrid(marg, marg)
    # leave only coords inside circular masks
    r      = np.sqrt(dx**2 + dy**2)
    # exclude center coordinate
    inloc  = np.logical_and(r <= R, r > 0)
    dx,dy  = dx[inloc],dy[inloc]
    # return a list of particle coords
    return np.int32([c + np.int32(zip(dx,dy)) for c in coords])

##### DEFINE WRITING DATA #############
class ParticleCoords2TFRecord(Directory2TFRecord):

    @staticmethod
    def preprocess_micro(microdata):
        micro,microdict = microdata
        D  = microdict['part_diameter']
        im = mrc.load(micro)[0]
        psize    = microdict['psize']
        szbn, bn = calc_micro_pick_shape(micro,D)
        psizebn  = psize * bn
        assert(np.all(np.int32(szbn)<np.int32(im.shape)))
        # resize image
        imbn  = cv2.resize(im, tuple(szbn[::-1]), interpolation=cv2.INTER_AREA)
        # remove bad pixels
        imbn  = image.unbad2D(imbn, thresh=5, neib=3)
        imbn  = image.float32_to_uint8(imbn)
        mrc.save(imbn, microdict['resizedname'], pixel_size=psizebn)

    def add_class_coords(self, allmicros, stars, cid):
        for starfile in stars:
            starfile = os.path.realpath(starfile)
            psize  = path2psize(starfile,'Select')
            D      = path2part_diameter(starfile)/psize
            params = parse_particles_star(starfile)
            for micro in params:
                if not micro in allmicros:
                    resizedname = os.path.join(self._resized_micros_dir, os.path.basename(micro))
                    allmicros.update({micro: {'coords': {cid: params[micro]['coords']}, 'ctf': params[micro]['ctf'],
                                              'part_diameter': D,'psize': psize,'resizedname':resizedname}})
                else:
                    if not cid in allmicros[micro]:
                        allmicros[micro]['coords'].update({cid: params[micro]['coords']})
                    else:
                        allmicros[micro]['coords'][cid].extend(params[micro]['coords'])

    def __init__(self,data_in_dir,data_out_dir):

        super(ParticleCoords2TFRecord, self).__init__(data_in_dir,data_out_dir)

        self._resized_micros_dir = os.path.join(ft.updirs(data_in_dir+'/',1), 'ResizedMicrographs')

        # each subdirectory in data_in_dir corresponds to a separate class
        # each class subdirectory contains symlink to a selection relion job
        topdirs = np.sort(os.walk(data_in_dir).next()[1])
        class2label = {}
        label2class = {}
        allmicros   = {}
        for d in range(len(topdirs)):
            cid = topdirs[d].tostring()
            tprint("Listing directory %s ..." % d)
            class2label.update({cid:d})
            label2class.update({d:cid})
            # get all star files with particle coordinates for this class
            stars = list_dirtree(os.path.join(data_in_dir,cid), 'particles.star')
            self.add_class_coords(allmicros,stars,cid)

        # initialize box class keys
        self.classes = class2label.keys()
        self.init_feature_keys(self.classes)
        # here allmicros has particle coordinates for each class per micrograph
        tprint("Removing particle overlaping in different classes ...")
        allmicros   = ParticleCoords2TFRecord.remove_class_overlap(allmicros)

        # count all particles
        classcnt    = np.zeros(len(class2label))
        for micro in allmicros:
            for cls in allmicros[micro]['coords']:
                classcnt[class2label[cls]] += len(allmicros[micro]['coords'][cls])
        # print totals
        # print "Total micrographs %d" % len(allmicros)
        for label in label2class:
            print 'Total coordinates in %s \t = %d' % (label2class[label],classcnt[label])

        if RESIZE_MICROS:
            print "Resizing %d micrographs ..." % len(allmicros)
            ft.mkdir_assure(self._resized_micros_dir)
            f = functools.partial(ParticleCoords2TFRecord.preprocess_micro)
            # Sequential
            # for micro in allmicros:
            #      f((micro,allmicros[micro]))
            # Parallel
            num_proc = 2 * mp.cpu_count()
            pool = ThreadPool(num_proc)
            pool.map(f, zip(allmicros.keys(), [allmicros[m] for m in allmicros]))

        # save updated micrographs
        self.allmicros = allmicros #dict(zip(allmicros.keys(),updmicros))

    @staticmethod
    def remove_class_overlap(allmicros):
        ''' remove particles that belong to more than one class '''
        for micro in allmicros:
            # available particle classes in the micrograph
            classes = allmicros[micro]['coords']
            D = allmicros[micro]['part_diameter'] #micro2part_diameter(micro))
            R = D / 2.

            # ------- add background coordinates around each particle ---------------
            if CLEAN in classes and AUGMENT_BACKGROUND:
                clean_coords = np.array(classes[CLEAN])
                addcoords = []
                for coord in clean_coords:
                    # add 4 neighbors
                    for dx in [-R, 0, R]:
                        for dy in [-R, 0, R]:
                            if (not (dx == 0 and dy == 0)) and (dx == 0 or dy == 0):
                                addcoords.append((coord + np.int32([dx, dy])).tolist())
                # append clean particle neighbors to background
                if BACKGROUND in classes:
                    classes[BACKGROUND] += addcoords
                else:
                    classes.update({BACKGROUND: addcoords})
            # -------------------------------------------------------------------------

            # collect all overlapping coordinates
            allcoords = np.zeros((0, 2))
            overlap = np.zeros((0, 2))
            for key in classes:
                coords = np.float32(classes[key])
                if allcoords.size > 0:
                    dist    = cdist(coords, allcoords).min(axis=1)
                    overlap = np.concatenate((overlap, coords[dist < D / 4., :]), 0)
                allcoords   = np.concatenate((allcoords, coords), 0)

            # remove all overlapping coordinates
            if overlap.size > 0:
                # remove only overlapping particles, but not bad/background and junk particles
                # for key in classes:
                key = CLEAN
                coords = np.float32(classes[key])
                dist   = cdist(coords, overlap).min(axis=1)
                classes[key] = np.delete(coords, np.where(dist < D / 4.)[0], axis=0).tolist()

                allmicros[micro]['coords'] = classes
        return allmicros

    ##### Overriding functions ########
    def test_example(self,provider):
        print "Generating example frames from the tfrecord ..."
        featsz  = (cfg.PICK_WIN//cfg.STRIDES[0],)*2
        out_dir = os.path.join(self.tfrecord_dir,'example_ground_truth')
        ft.rmtree_assure(out_dir)
        ft.mkdir_assure(out_dir)
        keys   = ['image']+self.classes+['dxyidxs','dxy']
        data   = provider.get(keys)
        with tf.Session().as_default() as sess:
            with tf.Graph().as_default():
                with tf.device('/device:CPU:0'):
                    coord = tf.train.Coordinator()
                    tf.train.start_queue_runners(coord=coord)
                    for i in range(N_EXAMPLES):
                        d = dict(zip(keys, sess.run(data)))
                        im    = d['image']
                        dd    = dict((k,unravel_coords_append(d[k],featsz)*cfg.STRIDES[0]) for k in self.classes)
                        plot_class_coords(im,dd,cfg.PART_D_PIXELS)
                        # save the resulting graph
                        fname = os.path.join(out_dir, 'example_%d' % i)
                        print "saving example %s, %d out of %d" % (fname,i,N_EXAMPLES)
                        savefig(plt.gcf(),fname)
                        # plt.gcf().set_figheight(10.0)
                        # plt.gcf().set_figwidth(10.0)
                        # fig = plt.gcf()
                        # fig.subplots_adjust(wspace=.1, hspace=0.2, left=0.03, right=0.98, bottom=0.05, top=0.93)
                        # fig.savefig(fname)
                        plt.close(plt.gcf())

    def init_feature_keys(self,box_keys=[]):
        self.feature_keys['byte_keys'] = list(box_keys) + [SHAPE_KEY,IMAGE_KEY] + ['dxyidxs','dxy']

    def calc_tile_coords(self,micro,x,y,bn):
        ''' calculates particle coordinates in the tile coordinate system '''
        # leave only particle that are at least particle diameter from the border
        allowed_border = cfg.PART_D_PIXELS
        tilesz  = (cfg.PICK_WIN,cfg.PICK_WIN)
        coords  = self.allmicros[micro]['coords']
        tcoords = {}
        for label in self.classes:
            if label in coords:
                lcoords = np.array(coords[label], dtype=np.float32)
                # switch x and y
                lcoords = lcoords[:,::-1]
                # convert global coords to binned
                lcoords = np.int32(np.round(lcoords/bn))
                # move to tile coordinates
                lcoords[:,0] -= x
                lcoords[:,1] -= y
                # leave only particles that are inside the tile
                loc_inside = (lcoords[..., 0] >= allowed_border) & \
                             (lcoords[..., 1] >= allowed_border) & \
                             (lcoords[..., 0] < tilesz[0] - allowed_border) & \
                             (lcoords[..., 1] < tilesz[1] - allowed_border)
                lcoords = lcoords[loc_inside]
            else:
                lcoords = np.zeros((0,2),dtype=np.int32)
            tcoords.update({label: lcoords})
        return tcoords

    def get_example_keys(self):
        tprint('Converting micrographs to tiles of %d pixels, particle diameter %d pixels ...' % (cfg.PICK_WIN,cfg.PART_D_PIXELS))
        micros = self.allmicros.keys()
        # random.shuffle(micros)
        # convert each micro into a set of tiles of cfg.PICK_WIN size
        # the training will run on one tile per record
        keys = []
        for m in micros:
            D     = self.allmicros[m]['part_diameter']
            sz,bn = calc_micro_pick_shape(m,D)
            xs,ys = image.tile2D(sz,(cfg.PICK_WIN,)*2,0.0)
            # create keys using tile coordinates
            for x in xs:
                for y in ys:
                    tcoords = self.calc_tile_coords(m,x,y,bn)
                    # see how many particles of each class
                    lens    = [len(tcoords[k]) for k in tcoords]
                    if np.sum(np.array(lens)) < 1:
                        # skip empty tile
                        continue
                    else:
                        keys.append('%s:%d,%d' % (m,x,y))

        # shuffle all example records to sit randomly in shards
        random.shuffle(keys)
        return keys

    def get_example_meta(self):
        return {'dtype': tf.uint8,
                'nclasses':len(self.classes),
                'classes':self.classes,
                'shape':(cfg.PICK_WIN,)*2}

    def key2byte_records(self,key):
        ''' Obtain a record with tile data '''

        # extract micrograph name and tile coordinates
        micro   = key.split(':')[0]
        x,y     = np.int32(key.split(':')[1].split(','))
        fname   = self.allmicros[micro]['resizedname']
        psize   = self.allmicros[micro]['psize']
        ctf     = self.allmicros[micro]['ctf']

        imbn,psizebn = mrc.load_psize(fname)
        imbn    = imbn[0]
        bn      = psizebn/psize
        # processing tile size
        tilesz  = (cfg.PICK_WIN,cfg.PICK_WIN)
        # the size of feature map
        featsz  = (cfg.PICK_WIN//cfg.STRIDES[0],)*2
        # particle diameter in feature image
        D       = cfg.PART_D_PIXELS//cfg.STRIDES[0]

        # cut tile out
        imt      = imbn[x:x+cfg.PICK_WIN,y:y+cfg.PICK_WIN]
        # flip phases
        imff     = ctf.phase_flip_cpu(imt,psizebn)
        # convert image to uint8
        im8      = image.float32_to_uint8(imff)
        # save image
        recs     = {IMAGE_KEY:im8.tobytes()}
        ## Save particle coordinates of the tile ##
        tcoords  = self.calc_tile_coords(micro,x,y,bn)

        # add coords of all particle types to the record
        for label in self.classes: #!!
            # convert coordinates to feature map dimensions
            lcoords = np.int32(np.round(np.float32(tcoords[label])/cfg.STRIDES[0]))
            recs.update({label: ravel_coords_append(lcoords, featsz).tobytes()})

            if label == CLEAN:
                # add coordinate corrections in the particle neighborhoods
                # get all particle coordinate around center coordinates
                dxycoords = neib_coords(lcoords, D)
                # get coordnate corrections
                dxy     = [p - c[None,:] for p,c in zip(dxycoords,lcoords)]
                dxyidxs = ravel_coords(dxycoords.reshape((-1,2)),featsz)
                dxy     = np.float32(dxy).reshape((-1,2))

                # obtain overlapping coordinates
                _,iidxs,counts = np.unique(dxyidxs,return_inverse=True,return_counts=True)
                novlp_loc   = counts[iidxs] == 1

                # leave ony nonoverlapping particle coordinate corrections dxy
                dxyidxs,dxy = dxyidxs[novlp_loc],dxy[novlp_loc]
                recs.update({'dxyidxs': np.append(dxyidxs,np.int64([0,0])).tobytes(),
                             'dxy':np.append(dxy,np.float32([-1,-1])).tobytes()})

        # save image shape
        recs.update({SHAPE_KEY:np.array(tilesz,dtype=np.int32).tobytes()})

        # image_data has to be in python str format
        return recs

    ##### DEFINE READING DATA ############
    @staticmethod
    def create_dataset(split_name, data_dir):
        return create_dataset(split_name, data_dir)

##
if __name__ == '__main__':

    # import numpy as np
    def pixel_size_2_io_box_sizes(psize,cls_psize=3.0,part_d=160):
        bin          = cls_psize/psize
        out_box_size = 1.5*part_d/cls_psize
        return np.round(bin*out_box_size),out_box_size

    # empty micrograph location
    #/jasper/ data / Livlab / autoprocess_cmm / empty_micrographs / Nucleosome_20170424_1821 / *.tbz
    print pixel_size_2_io_box_sizes(0.429109,3.0,160)
    # nonempty micrographs
    # /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20150925/*.tbz /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20150925_2/*.tbz /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20151208/*.tbz /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20151210/*.tbz
    print pixel_size_2_io_box_sizes(0.316704,3.0,160)


##   ###### GARBAGE #########

    # def part_d_from_jobfile(jobfile):
    #     '''Reads particle diameter from jobfile'''
    #     line = ft.get_line(jobfile, 'Particle diameter')
    #     return np.float32(line.split('==')[1])


    # def get_shapebn(self,shape,psize):
    #     bn     = psize2bn(psize)
    #     return tuple(np.int32(np.round(np.array(shape)*bn))) #.tolist()

    # chunks = utils.part_idxs(shardidxs,batch=num_proc)
    # for chunk in chunks:
    #     processes = []
    #     for idx in chunk:
    #         p = mp.Process(target=self.process_shard_single_proc, args=(keys[idx],))
    #         processes.append(p)
    #     # start processes
    #     [p.start() for p in processes]
    #     [p.join() for p in processes]
    #     sys.stdout.write('%s, image %d/%d' % (print_string, idx + 1, len(keys)))
    #     sys.stdout.flush()


    # def key2int_records(self,key):
    #     psize    = mrc.psize(key)
    #     shape    = mrc.shape(key)
    #     bn       = psize2bn(psize)
    #     coords   = self.allmicros[key]['coords']
    #     recs     = {}
    #     for label in coords.keys():
    #         lcoords  = np.array(coords[label],dtype=np.float32)
    #         # convert glbal coords to binned
    #         lcoords  = np.int32(np.round(bn*lcoords)).flatten().tolist()
    #         recs.update({label:lcoords})
    #     # add image size to int record
    #     szbn = list(self.get_shapebn(shape[1:],psize))
    #     recs.update({'shape':szbn})
    #     return recs


    # def float2uint16(im):
    #     ''' converts gain corrected and averaged counts to uint16 '''
    #     im  = im - im.min()
    #     im /= im.max()
    #     return np.uint16(512*im) #np.uint16(256*256*im)

    # def mrc2pngs(pngname,mrcname):
    #     ims = mrc.load(mrcname)
    #     for idx in range(ims.shape[0]):
    #         fname = pngname + '_%d.png' % idx
    #         if not os.path.exists(fname):
    #             png.from_array(float2uint16(ims[idx]),'L16').save(fname)

    # def convertmrcs2pngs(data_in_dir,pngdir):
    #     ft.mkdir_assure(pngdir)
    #     mrcnames = list_dirtree(data_in_dir, '.mrc')
    #     # length of input path
    #     inlen    = len(data_in_dir)
    #     # create png dir struct
    #     subdirs  = set([os.path.dirname(mrcname)[inlen:] for mrcname in mrcnames])
    #     for dir in subdirs:
    #         ft.mkdir_assure(os.path.join(pngdir,dir))
    #     # create png filename list
    #     pngnames = [os.path.join(pngdir,ft.replace_ext(name[inlen:],'')) for name in mrcnames]
    #     num_cores = multiprocessing.cpu_count()
    #     mrc2pngs(pngnames[0],mrcnames[0])
    #     print "Converting %d mrcs from %s, to pngs in %s, using %d cores" % \
    #           (len(mrcnames),data_in_dir,pngdir,num_cores)
    #     Parallel(n_jobs=2*num_cores)(delayed(mrc2pngs)(pngname,mrcname) for pngname,mrcname in zip(pngnames,mrcnames))

    # image_data  = tf.gfile.FastGFile(fname, 'r').read()
    # im    = misc.imread(fname)
    # #im    = misc.imresize(im,np.int32(np.float32(im.shape)/IMAGE_RESIZE),mode='F') #mtf.image.resize_images(im,tf.constant([180,240]))
    # #im    = np.uint16(im)

# from   fileio import mrc
# from   myplotlib import imshow
# im = mrc.load('//jasper/result/PKM2_WT/Extract/80_empty_empty_flip_38k/Movies//20170424_1821_A038_G004_H022_D001_avg.mrcs')
# imshow(im[0])