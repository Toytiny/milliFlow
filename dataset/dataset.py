import os
import sys
import glob
import numpy as np
import ujson
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class nDataset(Dataset):

    def __init__(self, args, root='/mnt/', partition='train', textio=None):

        self.npoints = args.num_points
        self.eval = args.eval
        self.partition = partition
        self.root = root + partition + '/'
        self.clips = os.listdir(self.root)
        self.samples = []
        self.clips_info = []
        num=0
        
        if args.model in ['pvraft', 'biflow', '3dflow']:
            self.eval = False
        
        for person_data in self.clips:
            root = os.path.join(self.root, person_data)
            clips = os.listdir(root)
            for clip in clips:
                labels = clip.split('_')
                label = labels[0] 
                if clip in ['head', 'squat', 'bow']:
                    continue
                
                clip_path = os.path.join(root, clip)
                samples = sorted(os.listdir(clip_path), key=lambda x: eval(x.split("_")[-1].split("-")[-1].split(".")[0]))

                if self.eval:
                    self.clips_info.append({'clip_name': clip,
                                            'index': [len(self.samples), len(self.samples) + len(samples)]
                                            })

                for j in range(len(samples)):
                    self.samples.append(os.path.join(clip_path, samples[j]))


    def __getitem__(self, index):

        with open(self.samples[index], 'rb') as fp:
            #print(self.samples[index])
            data = ujson.load(fp)

        data_1 = np.array(data["pc1"]).astype('float32')
        data_2 = np.array(data["pc2"]).astype('float32')

        # read input data and features
        pos_1 = data_1[:, 0:3]
        pos_2 = data_2[:, 0:3]
        feature_1 = data_1[:, 3:4]
        feature_2 = data_2[:, 3:4]
        labels = np.array(data["gt"]).astype('float32')


        feature_1 = np.insert(feature_1, 0, feature_1[:, 0], axis=1)
        feature_1 = np.insert(feature_1, 0, feature_1[:, 0], axis=1)
        feature_2 = np.insert(feature_2, 0, feature_2[:, 0], axis=1)
        feature_2 = np.insert(feature_2, 0, feature_2[:, 0], axis=1)


        ## downsample to npoints to enable fast batch processing (not in test)
        if not self.eval:

            npts_1 = pos_1.shape[0]
            npts_2 = pos_2.shape[0]

            ## if the number of points < npoints, fill empty space by duplicate sampling
            ##  (filler points less than 25%)
            # if npts_1 < self.npoints * 0.75:
            #    raise('the number of points is lower than {}'.format(self.npoints * 0.75))
            if npts_1 < self.npoints:
                sample_idx1 = np.arange(0, npts_1)
                sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1, self.npoints - npts_1, replace=True))
            else:
                sample_idx1 = np.random.choice(npts_1, self.npoints, replace=False)
            if npts_2 < self.npoints:
                sample_idx2 = np.arange(0, npts_2)
                sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2, self.npoints - npts_2, replace=True))
            else:
                sample_idx2 = np.random.choice(npts_2, self.npoints, replace=False)

            pos_1 = pos_1[sample_idx1, :]
            pos_2 = pos_2[sample_idx2, :]
            feature_1 = feature_1[sample_idx1, :]
            feature_2 = feature_2[sample_idx2, :]
            labels = labels[sample_idx1, :]

        return pos_1, pos_2, feature_1, feature_2, labels

    def __len__(self):
        return len(self.samples)