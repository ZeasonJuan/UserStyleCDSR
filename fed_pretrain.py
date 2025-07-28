import argparse
import numpy as np
import torch.nn as nn
from logging import getLogger
from recbole.config import Config
from recbole.data.dataloader import TrainDataLoader
from recbole.utils import init_seed, init_logger
from FLtrainer.fedtrainer_ldp import FedtrainTrainer
from model.vqrec import VQRec
from data.dataset import FederatedDataset
import os
import faiss
# import numpy as np
import torch

from utils import parse_faiss_index

def load_index(config, logger, item_num, field2id_token):
    code_dim = config['code_dim']
    code_cap = config['code_cap']
    dataset_name = config['dataset']
    index_suffix = config['index_suffix']
    if config['index_pretrain_dataset'] is not None:
        index_dataset = config['index_pretrain_dataset']
    else:
        index_dataset = config['dataset']
    index_path = os.path.join(
        config['index_path'],
        index_dataset,
        f'{index_dataset}.{index_suffix}'
    )
    logger.info(f'Index path: {index_path}')
    uni_index = faiss.read_index(index_path)
    pq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
    assert code_dim == pq_codes.shape[1], pq_codes.shape
    # assert item_num == 1 + pq_codes.shape[0], f'{item_num}, {pq_codes.shape}'
    # uint8 -> int32 to reserve 0 padding
    pq_codes = pq_codes.astype(np.int32)
    # 0 for padding
    pq_codes = pq_codes + 1
    # flatten pq codes
    base_id = 0
    for i in range(code_dim):
        pq_codes[:, i] += base_id
        base_id += code_cap + 1

    logger.info('Loading filtered index mapping.')
    filter_id_dct = {}
    with open(
            os.path.join(config['data_path'],
                         f'{dataset_name}.{config["filter_id_suffix"]}'),
            'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            filter_id_name = line.strip()
            filter_id_dct[filter_id_name] = idx

    logger.info('Converting indexes.')
    mapped_codes = np.zeros((item_num, code_dim), dtype=np.int32)
    for i, token in enumerate(field2id_token):
        if token == '[PAD]': continue
        mapped_codes[i] = pq_codes[filter_id_dct[token]]
    return torch.LongTensor(mapped_codes)

def change_dict(dict, point):
    for k in dict:
        if k == '[PAD]':
            continue
        dict[k] += point

def pretrain(dataset, **kwargs):
    # configurations initialization
    props = ['props/VQRec.yaml', 'props/pretrain.yaml']
    print(props)

    # configurations initialization
    config = Config(model=VQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    config_A = Config(model=VQRec, dataset='O', config_file_list=props, config_dict=kwargs)
    config_B = Config(model=VQRec, dataset='A', config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    init_seed(config_A['seed'], config['reproducibility'])
    init_seed(config_B['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    init_logger(config_A)
    init_logger(config_B)
    logger = getLogger()
    logger.info(config)
    logger.info(config_A)
    logger.info(config_B)
    logger.info(dataset)

    dataset_A = FederatedDataset(config_A, pq_codes=None)
    logger.info(dataset_A)
    #这个是得到训练集，按照Recbole默认的config中的eval_setting设置
    #有个问题，默认的eval_setting是随机顺序、按比例切分、全量排名
    pretrain_dataset_A = dataset_A.build()[0]
    #三部分：1.得到所有(raw_id, token_id)的元组。2.列表化并取最后一个元素。3.取第二个元素，即token_id
    spilt_point = list(dataset_A.field2token_id['item_id'].items())[-1][1]
    dataset_B = FederatedDataset(config_B, pq_codes=None)
    # B_field2id_token = copy.deepcopy(dataset_B.field2token_id['item_id'])
    # change_dict(B_field2id_token, spilt_point)
    # change_dict(dataset_B.field2token_id['item_id_list'], spilt_point)
    # dataset_B.inter_feat.item_id += spilt_point
    # dataset_B.inter_feat.item_id_list += spilt_point

    logger.info(dataset_B)
    pretrain_dataset_B = dataset_B.build()[0]
    item_num = dataset_A.item_num + dataset_B.item_num - 1
    field2id_token = np.concatenate((dataset_A.field2id_token['item_id'], dataset_B.field2id_token['item_id'][1:]))
    #这里得到了两个数据集合并的原始item_id列表field2id_token
    pq_codes = load_index(config, logger, item_num, field2id_token).to(config['device'])
    item_pq_A = pq_codes[:spilt_point + 1]
    item_pq_B = torch.cat([pq_codes[0].unsqueeze(0), pq_codes[spilt_point + 1:]], dim=0)
    pretrain_dataset_A.pq_codes = item_pq_A
    pretrain_dataset_B.pq_codes = item_pq_B
    pretrain_data_A = TrainDataLoader(config_A, pretrain_dataset_A, None, shuffle=True)
    pretrain_data_B = TrainDataLoader(config_A, pretrain_dataset_B, None, shuffle=True)

    model_A = VQRec(config_A, pretrain_data_A.dataset).to(config['device'])
    model_A.pq_codes.to(config['device'])
    logger.info(model_A)
    model_B = VQRec(config_B, pretrain_data_B.dataset).to(config['device'])
    model_B.pq_codes.to(config['device'])
    logger.info(model_B)
    global_embedding = nn.Embedding(
        config['code_dim'] * (1 + config['code_cap']), config['hidden_size'], padding_idx=0).to(config['device'])
    global_embedding.weight.data.normal_(mean=0.0, std=config['initializer_range'])
    model_A.pq_code_embedding.load_state_dict(global_embedding.state_dict())
    model_B.pq_code_embedding.load_state_dict(global_embedding.state_dict())
    weight = []
    weight.append(dataset_A.file_size_list[0]/(dataset_A.file_size_list[0] + dataset_B.file_size_list[0]))
    weight.append(dataset_B.file_size_list[0]/(dataset_A.file_size_list[0] + dataset_B.file_size_list[0]))
    weight = torch.tensor(weight).to(config['device'])
    trainer = FedtrainTrainer(config_A, config_B, model_A, model_B, global_embedding)
    trainer.fedtrain(pretrain_data_A, pretrain_data_B, weight, show_progress=True)

    return config['model'], config['dataset']


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-d', type=str, default='OA', help='dataset name')
    # parser.add_argument('--epsilon', type=float, default=eps, help='LDP epsilon')
    # parser.add_argument('--num_bits', type=int, default=bits, help='number of PQ bits')
    args, _ = parser.parse_known_args()
    # Call pretrain with these hyperparameters
    model, dataset = pretrain(args.d)




   
