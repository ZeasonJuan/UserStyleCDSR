import os
import faiss
import numpy as np
import torch
from recbole.data.dataset import SequentialDataset
import os.path as osp
import torch.nn as nn

from utils import parse_faiss_index


class VQRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = self.load_index()

    def load_index(self):
        if self.config['index_pretrain_dataset'] is not None:
            index_dataset = self.config['index_pretrain_dataset']
        else:
            index_dataset = self.dataset_name
        index_path = os.path.join(
            self.config['index_path'],
            index_dataset,
            f'{index_dataset}.{self.index_suffix}'
        )
        self.logger.info(f'Index path: {index_path}')
        uni_index = faiss.read_index(index_path)
        old_pq_codes, _, _, _ = parse_faiss_index(uni_index)
        old_code_num = old_pq_codes.shape[0]

        self.plm_suffix = self.config['plm_suffix']
        self.plm_size = self.config['plm_size']
        feat_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        uni_index.add(loaded_feat)
        all_pq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
        pq_codes = all_pq_codes[old_code_num:]
        assert self.code_dim == pq_codes.shape[1], pq_codes.shape
        assert self.item_num == 1 + pq_codes.shape[0], pq_codes.shape

        # uint8 -> int32 to reserve 0 padding
        pq_codes = pq_codes.astype(np.int32)
        # 0 for padding
        pq_codes = pq_codes + 1
        # flatten pq codes
        base_id = 0
        for i in range(self.code_dim):
            pq_codes[:, i] += base_id
            base_id += self.code_cap + 1

        mapped_codes = np.zeros((self.item_num, self.code_dim), dtype=np.int32)
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_codes[i] = pq_codes[int(token)]
            
        self.plm_embedding = torch.FloatTensor(loaded_feat)
        return torch.LongTensor(mapped_codes)


class PretrainVQRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = self.load_index()

    def load_index(self):
        if self.config['index_pretrain_dataset'] is not None:
            index_dataset = self.config['index_pretrain_dataset']
        else:
            index_dataset = self.dataset_name
        index_path = os.path.join(
            self.config['index_path'],
            index_dataset,
            f'{index_dataset}.{self.index_suffix}'
        )
        self.logger.info(f'Index path: {index_path}')
        uni_index = faiss.read_index(index_path)
        pq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
        assert self.code_dim == pq_codes.shape[1], pq_codes.shape
        # assert self.item_num == 1 + pq_codes.shape[0], f'{self.item_num}, {pq_codes.shape}'

        # uint8 -> int32 to reserve 0 padding
        pq_codes = pq_codes.astype(np.int32)
        # 0 for padding
        pq_codes = pq_codes + 1
        # flatten pq codes
        base_id = 0
        for i in range(self.code_dim):  # 给每个item的每一位加上257，表示从257开始，这样是因为一方面需要每个code的表示需要在一个code embedding里面查找，另一方面是32位的每个维度为256的表是不一样
            pq_codes[:, i] += base_id
            base_id += self.code_cap + 1

        self.logger.info('Loading filtered index mapping.')
        filter_id_dct = {}
        with open(
            os.path.join(self.config['data_path'],
                f'{self.dataset_name}.{self.config["filter_id_suffix"]}'),
            'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                filter_id_name = line.strip()
                filter_id_dct[filter_id_name] = idx
            
        self.logger.info('Converting indexes.')
        mapped_codes = np.zeros((self.item_num, self.code_dim), dtype=np.int32)
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_codes[i] = pq_codes[filter_id_dct[token]]
        return torch.LongTensor(mapped_codes)

class FederatedDataset(SequentialDataset):
    def __init__(self, config, pq_codes):
        super().__init__(config)
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = pq_codes

class FederatedPlusDataset(SequentialDataset):
    def __init__(self, config, pq_codes, pq_codes_user=None):
        super().__init__(config)
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = pq_codes
        self.pq_codes_user = pq_codes_user

class FederatedDatasetRQ(SequentialDataset):
    def __init__(self, config, text_embedding):
        super().__init__(config)
        self.text_embedding = None

class FdsaDataset(SequentialDataset):
    def __init__(self, config, pq_codes):
        super().__init__(config)
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = pq_codes
        self.private_codes = self.load_index()

    def load_index(self):
        if self.config['index_pretrain_dataset'] is not None:
            index_dataset = self.config['index_pretrain_dataset']
        else:
            index_dataset = self.dataset_name
        index_path = os.path.join(
            self.config['index_path'],
            index_dataset,
            f'{index_dataset}.{self.index_suffix}'
        )
        self.logger.info(f'Index path: {index_path}')
        uni_index = faiss.read_index(index_path)
        pq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
        assert self.code_dim == pq_codes.shape[1], pq_codes.shape
        # assert self.item_num == 1 + pq_codes.shape[0], f'{self.item_num}, {pq_codes.shape}'

        # uint8 -> int32 to reserve 0 padding
        pq_codes = pq_codes.astype(np.int32)
        # 0 for padding
        pq_codes = pq_codes + 1
        # flatten pq codes
        base_id = 0
        for i in range(
                self.code_dim):  # 给每个item的每一位加上257，表示从257开始，这样是因为一方面需要每个code的表示需要在一个code embedding里面查找，另一方面是32位的每个维度为256的表是不一样
            pq_codes[:, i] += base_id
            base_id += self.code_cap + 1

        self.logger.info('Loading filtered index mapping.')
        filter_id_dct = {}
        with open(
                os.path.join(self.config['data_path'],
                             f'{self.dataset_name}.{self.config["filter_id_suffix"]}'),
                'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                filter_id_name = line.strip()
                filter_id_dct[filter_id_name] = idx

        self.logger.info('Converting indexes.')
        mapped_codes = np.zeros((self.item_num, self.code_dim), dtype=np.int32)
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_codes[i] = pq_codes[filter_id_dct[token]]
        return torch.LongTensor(mapped_codes)


class UniVQRecDataset(SequentialDataset):
    def __init__(self, config, pq_codes):
        super().__init__(config)

        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        self.pq_codes = pq_codes
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_index(self):
        if self.config['index_pretrain_dataset'] is not None:
            index_dataset = self.config['index_pretrain_dataset']
        else:
            index_dataset = self.dataset_name
        index_path = os.path.join(
            self.config['index_path'],
            index_dataset,
            f'{index_dataset}.{self.index_suffix}'
        )
        self.logger.info(f'Index path: {index_path}')
        uni_index = faiss.read_index(index_path)
        pq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
        assert self.code_dim == pq_codes.shape[1], pq_codes.shape
        # assert self.item_num == 1 + pq_codes.shape[0], f'{self.item_num}, {pq_codes.shape}'

        # uint8 -> int32 to reserve 0 padding
        pq_codes = pq_codes.astype(np.int32)
        # 0 for padding
        pq_codes = pq_codes + 1
        # flatten pq codes
        base_id = 0
        for i in range(
                self.code_dim):  # 给每个item的每一位加上257，表示从257开始，这样是因为一方面需要每个code的表示需要在一个code embedding里面查找，另一方面是32位的每个维度为256的表是不一样
            pq_codes[:, i] += base_id
            base_id += self.code_cap + 1

        self.logger.info('Loading filtered index mapping.')
        filter_id_dct = {}
        with open(
                os.path.join(self.config['data_path'],
                             f'{self.dataset_name}.{self.config["filter_id_suffix"]}'),
                'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                filter_id_name = line.strip()
                filter_id_dct[filter_id_name] = idx

        self.logger.info('Converting indexes.')
        mapped_codes = np.zeros((self.item_num, self.code_dim), dtype=np.int32)
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_codes[i] = pq_codes[filter_id_dct[token]]
        return torch.LongTensor(mapped_codes)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token.split('-')[1])]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding

class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding