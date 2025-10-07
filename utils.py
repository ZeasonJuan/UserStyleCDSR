import importlib
import os
import faiss
from recbole.data.utils import create_dataset as create_recbole_dataset
import numpy as np
import torch
import pickle

def parse_faiss_index(pq_index):
    vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
    assert isinstance(vt, faiss.LinearTransform)
    opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

    ivf_index = faiss.downcast_index(pq_index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)
    
    centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
    centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
    
    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.rev_swig_ptr(coarse_quantizer.get_xb(), ivf_index.pq.M * ivf_index.pq.dsub)
    coarse_embeds = coarse_embeds.reshape(-1)

    return pq_codes, centroid_embeds, coarse_embeds, opq_transform


def create_dataset(config):
    dataset_module = importlib.import_module('data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        return getattr(dataset_module, config['model'] + 'Dataset')(config)
    else:
        return create_recbole_dataset(config)


class Seq2BertBank:
    def __init__(self, config, dataset_all_name, l2norm=False, mmap=True):
        # 1) 读 npy（用 memmap 节省内存）
        device = config['device']
        self.OUTPUT_FILE_PATH = os.path.join(config['data_path'], f"{dataset_all_name}.torch2numpy_dict.pkl")
        map_tsv = os.path.join(config['data_path'],
                                 f"{dataset_all_name}.seq2bert_train.map.tsv")
        npy_path = os.path.join(config['data_path'],
                                 f"{dataset_all_name}.seq2bert_train.npy")
        map_tsv_test = os.path.join(config['data_path'],
                                 f"{dataset_all_name}.seq2bert_test.map.tsv")
        npy_path_test = os.path.join(config['data_path'],
                                 f"{dataset_all_name}.seq2bert_test.npy")
        map_tsv_valid = os.path.join(config['data_path'],
                                 f"{dataset_all_name}.seq2bert_valid.map.tsv")
        npy_path_valid = os.path.join(config['data_path'],
                                 f"{dataset_all_name}.seq2bert_valid.npy")
        self.vecs = np.load(npy_path, mmap_mode="r" if mmap else None)  # (N, D)
        self.vecs_test = np.load(npy_path_test, mmap_mode="r" if mmap else None)  # (N, D)
        self.vecs_valid = np.load(npy_path_valid, mmap_mode="r" if mmap else None)  # (N, D)
        print(f"Loaded seq2bert bank from {npy_path} with shape {self.vecs.shape}, "
                f"and {npy_path_test} with shape {self.vecs_test.shape}, "
                f"and {npy_path_valid} with shape {self.vecs_valid.shape}.")

        self.D = self.vecs.shape[1]
        self.device = torch.device(device)
        self.l2norm = l2norm
        self.id_dict = None
        # key是训练时的torch.Tensor(1维)，value是对应的embedding（1维的np.array）
        # try: 
        #     with open(self.OUTPUT_FILE_PATH, "rb") as f:
        #         loaded_data = pickle.load(f)
        #         self.train_torch_seq_to_emb = loaded_data
        # except FileNotFoundError:
        self.train_torch_seq_to_emb = {}

        # 2) 读映射
        self.key2row = {}
        with open(map_tsv, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # 支持一种格式：
                # a) row_id \t seq
                parts = line.rstrip("\n").split("\t")
                if parts[0] == "row_index": 
                    continue  # 跳过表头
                tokens = parts[1].strip().split()
                str_seq = " ".join(str(int(tok.split("-")[-1])) for tok in tokens if tok)
                key = str_seq
                row = int(parts[0])
                self.key2row[key] = row
        self.key2row_test = {}
        with open(map_tsv_test, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # 支持一种格式：
                # a) row_id \t seq
                parts = line.rstrip("\n").split("\t")
                if parts[0] == "row_index": 
                    continue  # 跳过表头
                tokens = parts[1].strip().split()
                str_seq = " ".join(str(int(tok.split("-")[-1])) for tok in tokens if tok)
                key = str_seq
                row = int(parts[0])
                self.key2row_test[key] = row
        self.key2row_valid = {}
        with open(map_tsv_valid, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # 支持一种格式：
                # a) row_id \t seq
                parts = line.rstrip("\n").split("\t")
                if parts[0] == "row_index": 
                    continue  # 跳过表头
                tokens = parts[1].strip().split()
                str_seq = " ".join(str(int(tok.split("-")[-1])) for tok in tokens if tok)
                key = str_seq
                row = int(parts[0])
                self.key2row_valid[key] = row

    def save_to_pickle(self):
        with open(self.OUTPUT_FILE_PATH, 'wb') as f:
            pickle.dump(self.train_torch_seq_to_emb, f)
        print(f'Saved train_torch_seq_to_emb to {self.OUTPUT_FILE_PATH}, size: {len(self.train_torch_seq_to_emb)}')


    def get_batch(self, batch_seqs, training=True):
        """batch_seqs: 二维的torch.Tensor"""
        assert isinstance(batch_seqs, torch.Tensor) and batch_seqs.dim() == 2
        key2row_test = self.key2row_test
        vecs_test = self.vecs_test
        key2row_valid = self.key2row_valid
        vecs_valid = self.vecs_valid
        key2row = self.key2row
        vecs = self.vecs
        
        out = np.zeros((len(batch_seqs), self.D), dtype=np.float32)
        index = 0
        for s in batch_seqs:
            key = s
            if key in self.train_torch_seq_to_emb.keys():
                out[index] = self.train_torch_seq_to_emb[key]
            else: 
                idx = (key != 0).nonzero(as_tuple=True)[0]    
                lst = key[:idx[-1]+1].tolist()
                str_list = " ".join(str(self.id_dict[x]) for x in lst)
                row_index = key2row.get(str_list, -1)
                if row_index == -1: # 训练集中没有，去测试集中找    
                    row_index = key2row_test.get(str_list, -1)
                    if row_index == -1:
                        row_index = key2row_valid.get(str_list, -1)
                        if row_index == -1:
                            print('not found seq: ', str_list)
                            raise ValueError("seq not found in seq2bert bank")
                        out[index] = vecs_valid[row_index]
                    else:
                        out[index] = vecs_test[row_index]
                else:
                    out[index] = vecs[row_index]
                
                self.train_torch_seq_to_emb[key] = out[index] # 记住这个seq的embedding
            index += 1
        return out
