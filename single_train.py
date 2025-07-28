import argparse
import numpy as np
from logging import getLogger
from recbole.config import Config
from recbole.data.dataloader import TrainDataLoader
from recbole.utils import set_color
from recbole.utils import init_seed, init_logger
# from model.vqrec_attprompt import VQRec
from model.vqrec import VQRec
# from model.vqrec_tid import VQRec
# from vqrec import VQRec
from data.dataset import FederatedDataset
from data.dataset import FdsaDataset
from data.dataset import UniVQRecDataset
from sklearn.model_selection import ParameterGrid
import os
import faiss
import torch
from recbole.data import data_preparation
from trainer import VQRecTrainer
from utils import parse_faiss_index

def load_index(config, logger, field2id_token, item_num):
    code_dim = config['code_dim']
    code_cap = config['code_cap']
    dataset_name = config['dataset']
    index_suffix = config['index_suffix']
    pq_path = config['pq_data']
    if config['index_pretrain_dataset'] is not None:
        index_dataset = config['index_pretrain_dataset']
    else:
        index_dataset = config['dataset']
    index_path = os.path.join(
        config['index_path'],
        pq_path,
        f'{pq_path}.{index_suffix}'
    )
    logger.info(f'Index path: {index_path}')
    uni_index = faiss.read_index(index_path)
    pq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
    assert code_dim == pq_codes.shape[1], pq_codes.shape
    # uint8 -> int32 to reserve 0 padding
    pq_codes = pq_codes.astype(np.int32)
    # 0 for padding
    pq_codes = pq_codes + 1
    # flatten pq codes
    base_id = 0
    for i in range(code_dim):  # 给每个item的每一位加上257，表示从257开始，这样是因为一方面需要每个code的表示需要在一个code embedding里面查找，另一方面是32位的每个维度为256的表是不一样
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

def finetune(model_name, dataset, pretrained_file='', finetune_mode='', **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=VQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    config_A = Config(model=VQRec, dataset='O', config_file_list=props, config_dict=kwargs) #这里的P要进行设置，是要进行微调的单个数据集的首字母缩写
    init_seed(config['seed'], config['reproducibility'])
    init_seed(config_A['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config_A)
    logger = getLogger()
    logger.info(config_A)
    dataset = FederatedDataset(config_A, pq_codes=None)
    dataset_A = FederatedDataset(config_A, pq_codes=None)
    logger.info(dataset_A)
    pretrain_dataset_A = dataset_A.build()[0]
    pq_codes = load_index(config, logger, dataset_A.field2id_token['item_id'], dataset_A.item_num).to(config['device'])
    pretrain_dataset_A.pq_codes = pq_codes
    dataset.pq_codes = pq_codes
    pretrain_data_A = TrainDataLoader(config_A, pretrain_dataset_A, None, shuffle=True)
    train_data, valid_data, test_data = data_preparation(config_A, dataset)

    model_A = VQRec(config_A, pretrain_data_A.dataset).to(config['device'])
    model_A.pq_codes.to(config['device'])
    # model_A.private_codes = model_A.private_codes.to(config['device'])
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model_A.load_state_dict(checkpoint['state_dict'], strict=False)
        if finetune_mode == 'fix_enc':
            logger.info('[Fine-tune mode] Fix Seq Encoder!')
            for _ in model_A.position_embedding.parameters():
                _.requires_grad = False
            for _ in model_A.trm_encoder.parameters():
                _.requires_grad = False
            for _ in model_A.pq_code_embedding.parameters():
                _.requires_grad = False
        elif finetune_mode == 'fix_prompt':
            logger.info('[Fine-tune mode] Fix Prompt!')
            model_A.prompts.requires_grad = False
            for _ in model_A.attn_layer.parameters():
                _.requires_grad = False
    logger.info(model_A)
    trainer = VQRecTrainer(config_A, model_A)
    best_valid_score, best_valid_result = trainer.fit(pretrain_data_A, valid_data, show_progress=True)
    # best_valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=True)
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')

    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config_A['model'], config_A['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config_A['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
#同预训练下改微调数据集，全量训练分三步：

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='VQRec', help='model name')
    parser.add_argument('-d', type=str, default='OA', help='dataset name') # 😁
    # OP 😁
    # parser.add_argument('-p', type=str, default='save_OP/VQRec-P-10-2025-06-29-e0.3-nb8.pth', help='pre-trained model path')
    # parser.add_argument('-p', type=str, default='save_OP/VQRec-O-10-2025-06-29-e0.3-nb8.pth', help='pre-trained model path')
    # OA 
    # parser.add_argument('-p', type=str, default='save_OA/VQRec-A-10-2025-07-10.pth', help='pre-trained model path')
    parser.add_argument('-p', type=str, default='save_OA/VQRec-O-10-2025-07-10.pth', help='pre-trained model path')
    parser.add_argument('-f', type=str, default='', help='fine-tune mode')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.m, args.d, pretrained_file=args.p)

