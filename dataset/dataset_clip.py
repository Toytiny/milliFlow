import os
import sys
import glob
import numpy as np
import ujson
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

class ClipDataset(Dataset):

    def __init__(self, args, root='', partition='train', textio=None):

        self.npoints = args.num_points
        self.eval = args.eval
        self.partition = partition
        self.root = root + partition + '/'
        self.mini_clip_len = args.mini_clip_len
        self.update_len = args.update_len
        self.person = os.listdir(self.root)

        self.mini_samples = []
        self.samples = []
        self.clips_info = []
        self.mini_clips_info = []

        for p in self.person:
            clips_path = os.path.join(self.root, p)
            clips = os.listdir(clips_path)
            for clip in clips:
                clip_path = os.path.join(clips_path, clip)
                samples = sorted(os.listdir(clip_path), key=lambda x: eval(x.split("_")[-1].split("-")[-1].split(".")[0]))

                if self.eval:
                    self.clips_info.append({'clip_name': clip,
                                            'index': [len(self.samples), len(self.samples) + len(samples)]
                                            })

                    for j in range(len(samples)):
                        self.samples.append(os.path.join(clip_path, samples[j]))

                if not self.eval:
                    clip_num = int(np.floor(len(samples) / self.mini_clip_len))
                    ## take mini_clip as a sample
                    for i in range(clip_num):
                        st_idx = i * self.mini_clip_len
                        mini_sample = []
                        for j in range(self.mini_clip_len):
                            mini_sample.append(os.path.join(clip_path, samples[st_idx + j]))
                            self.samples.append(os.path.join(clip_path, samples[st_idx + j]))
                        self.mini_samples.append(mini_sample)

        """
        if not self.eval:
            self.textio.cprint(self.partition + ' : ' + str(len(self.mini_samples)) + ' mini_clips')
        if self.eval:
            self.textio.cprint(self.partition + ' : ' + str(len(self.samples)) + ' frames')
        """

    def __getitem__(self, index):
        if not self.eval:
            return self.get_clip_item(index)
        if self.eval:
            with open(self.samples[index], 'rb') as fp:
                data = ujson.load(fp)

            return self.get_sample_item(data)

    def get_sample_item(self, data, mini_sample):

        data_1 = np.array(data["pc1"]).astype('float32')
        data_2 = np.array(data["pc2"]).astype('float32')

        if len(data_1.shape) !=2:
            print(mini_sample)
        # read input data and features
        pos_1 = data_1[:, 0:3]
        pos_2 = data_2[:, 0:3]
        feature_1 = data_1[:, 3:4]
        feature_2 = data_2[:, 3:4]
        labels = np.array(data["gt"]).astype('float32')
        pos_labels = data_1[:, 4:5]

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
            pos_labels = pos_labels[sample_idx1, :]

        return pos_1, pos_2, feature_1, feature_2, labels

    def get_clip_item(self, index):
        mini_sample = self.mini_samples[index]
        mini_pos_1 = np.zeros((self.mini_clip_len, self.npoints, 3)).astype('float32')
        mini_pos_2 = np.zeros((self.mini_clip_len, self.npoints, 3)).astype('float32')
        mini_feat_1 = np.zeros((self.mini_clip_len, self.npoints, 3)).astype('float32')
        mini_feat_2 = np.zeros((self.mini_clip_len, self.npoints, 3)).astype('float32')
        mini_labels = np.zeros((self.mini_clip_len, self.npoints, 3)).astype('float32')
        mini_pos_labels = np.zeros((self.mini_clip_len, self.npoints, 3)).astype('float32')

        for i in range(0, len(mini_sample)):
            with open(mini_sample[i], 'rb') as fp:
                data = ujson.load(fp)

            pos_1, pos_2, feature_1, feature_2, labels = self.get_sample_item(data,mini_sample)

            # accumulate sample information
            mini_pos_1[i] = pos_1
            mini_pos_2[i] = pos_2
            mini_feat_1[i] = feature_1
            mini_feat_2[i] = feature_2
            mini_labels[i] = labels

        return mini_pos_1, mini_pos_2, mini_feat_1, mini_feat_2, mini_labels

    def __len__(self):
        if not self.eval:
            return len(self.mini_samples)
        if self.eval:
            return len(self.samples)