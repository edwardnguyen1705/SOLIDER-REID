# encoding: utf-8
"""
"""
import os
from glob import glob
from tqdm import tqdm
import os.path as osp
import random

from .bases import BaseImageDataset

IMG_EXT = "jpg"

class CCDMMPSSSSS(BaseImageDataset):
    """CCDMMPSSSSS
    """
    dataset_dir = ""
    dataset_name = 'ccdmmpsssss'
    splits = "splits"

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(CCDMMPSSSSS, self).__init__()
        self.root = root
        self.data_bdir = osp.join(self.root, self.dataset_dir, self.dataset_name)
        
        self.train_dir = osp.join(self.data_bdir, 'train')
        self.query_dir = osp.join(self.data_bdir, 'query')
        self.gallery_dir = osp.join(self.data_bdir, 'gallery')
        
        self.im_selected_prob = {"train": 1.0, "query": 0.25, "gallery": 0.25}

        self.__pid2label = None
        self.pid_begin = pid_begin
        
        train = self._process_dir(self.train_dir, split="train")
        gallery = self._process_dir(self.gallery_dir, split="gallery")
        query = self._process_dir(self.query_dir, split="query")

        if verbose:
            print("=> CCDMMPSSSSS loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        

    def _process_dir(self, split_dir, split="train"):
        random.seed(42) # make exps reproducible
        pid_container = set()
        img_paths = []
        for root, dirs, files in os.walk(split_dir):
            for pid in dirs:
                pid_container.add(pid)
                pid_dir = osp.join(split_dir, pid)
                img_paths_ = glob(osp.join(pid_dir, f"*.{IMG_EXT}"))
                for img_p in img_paths_:
                    if random.random() > self.im_selected_prob[split]: continue
                    camid = osp.basename(img_p).replace(pid+"_", "").split("_")[0]
                    img_paths.append((img_p, pid, int(camid)))

        if split != "query":
            self.__pid2label = {pid: label for label, pid in enumerate(pid_container)}
            
        # assumption
        # order: train, g, q
        # query is a subset of galley, else this `self.__pid2label[pid]` returns key err
        if split != "query":
            self.__pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        pid_container_test = set()
        dataset = []
        
        for fp, pid_str, camid in tqdm(img_paths):
            pid = self.__pid2label.get(pid_str, None)
            if pid is None: continue
            pid_container_test.add(pid)
            dataset.append((fp, self.pid_begin + pid, camid, 1))
        
        print(f"split: {split}, num of pids: {len(pid_container_test)}")
        # check if pid starts from 0 and increments with 1
        if split == "train":
            for idx, pid in enumerate(pid_container_test):
                if idx == pid: continue
                print(idx, pid)
                assert idx == pid, "See code comment for explanation"
        
        return dataset
