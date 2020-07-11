import os
import cv2
import numpy as np
from PIL import Image

from dict_map import ClassMap

class DataProvider:

    def __init__(self, data_list, need_labels=True, is_shuffle=False):
        self._data_list = data_list
        self._need_labels = need_labels
        self._is_shuffle = is_shuffle
        self._cur_i = 0
        self._classmap = ClassMap()

    def size(self):
        return len(self._data_list)

    def __call__(self, n):
        return self._load_data(n)


    def _load_data(self, n):
        imgs = []
        labs = []
        for _ in range(n):
            data_path = self._data_list[self._cur_i]
            img = Image.open(data_path)
            img = self._pre_processing(img)
            imgs.append(img)

            if self._need_labels:
                lab_str = data_path.split(os.sep)[-2]
                lab_idx = self._classmap.get_idx(lab_str)
                # one hot
                lab = np.eye(15)[lab_idx]
                labs.append(lab)
            self._next_idx(1)

        return np.array(imgs, np.float32), np.array(labs, np.float32)

    def _pre_processing(self, img):
        # convert image to numpy.ndarray
        img = np.array(img)

        # resize to 224 * 224
        img = cv2.resize(img, (224,224))

        # zero mean
        img = (img - np.mean(img)) / np.std(img)

        # extend dimensions for the gray scaled image in order to pass it to tensorflow (x, y, c)
        img = np.expand_dims(img, -1)
        return img

    def _next_idx(self, n=1):
        # Cycle index.
        self._cur_i += n
        if self._cur_i >= len(self._data_list):
            self._cur_i = self._cur_i % len(self._data_list)
            if self._is_shuffle:
                np.random.shuffle(self._data_list)
