import os
import argparse
import numpy as np
import faiss
from tqdm import tqdm
from recbole.data.dataset import SequentialDataset


def parse_args():
    # Basic
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='OR,Pantry')
    # parser.add_argument('--input_path', type=str, default='dataset/or-pantry/')
    # parser.add_argument('--output_path', type=str, default='dataset/or-pantry/')
    parser.add_argument('--dataset', type=str, default='Office,Arts')
    parser.add_argument('--input_path', type=str, default='dataset/office-arts/')
    parser.add_argument('--output_path', type=str, default='dataset/office-arts/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--suffix', type=str, default='feat1CLS')
    parser.add_argument('--plm_size', type=int, default=768)

    # PQ
    parser.add_argument("--subvector_num", type=int, default=48, help='16/24/32/48/64/96')
    parser.add_argument("--n_centroid", type=int, default=8)
    parser.add_argument("--use_gpu", type=int, default=False)
    parser.add_argument("--strict", type=int, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    dataset_names = args.dataset.split(',')
    print('Convert dataset: ')
    print(' Dataset: ', dataset_names)

    short_name = ''.join([_[0] for _ in dataset_names])
    print(' Short name: ', short_name)
    #严格模式下，加载训练集和测试集的交集
    #严格模式，将训练集与测试集的交集存入filter_id这个文件中
    if args.strict:
        item_set = [set() for i in range(len(dataset_names))]
        user_dict = [dict() for i in range(len(dataset_names))]
        user_set = [set() for i in range(len(dataset_names))]
        inter_path = os.path.join(args.input_path, short_name, f'{short_name}.test.inter')
        print(f'Strict Mode: Loading training data from [{inter_path}]')
        with open(inter_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in tqdm(file):
                user_id, item_seq, item_id = line.strip().split('\t')
                did, pure_item_id = item_id.split('-')
                items_in_seq = item_seq.split(' ')
                _, pure_user_id = user_id.split('-')
                pure_user_id = int(pure_user_id)
                assert _ == did
                pure_items_in_seq = [int(_.split('-')[-1]) for _ in items_in_seq]
                # if pure_user_id not in user_dict[int(did)].keys(): 
                #     user_dict[int(did)][int(pure_user_id)] = pure_items_in_seq
                # else:
                #     #如果用户的序列长度小于当前的序列长度，则更新
                #     if len(pure_items_in_seq) > len(user_dict[int(did)][int(pure_user_id)]):
                #         print('pure_user_id: ', pure_user_id, 'updated seq: ', user_dict[int(did)][int(pure_user_id)], 'to', pure_items_in_seq)
                #         user_dict[int(did)][int(pure_user_id)] = pure_items_in_seq
                user_set[int(did)].add(int(pure_user_id))
        
        inter_path = os.path.join(args.input_path, short_name, f'{short_name}.train.inter')
        with open(inter_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in tqdm(file):
                user_id, item_seq, item_id = line.strip().split('\t')
                did, pure_item_id = item_id.split('-')
                items_in_seq = item_seq.split(' ')
                _, pure_user_id = user_id.split('-')
                pure_user_id = int(pure_user_id)
                assert _ == did
                pure_items_in_seq = [int(_.split('-')[-1]) for _ in items_in_seq]
                if pure_user_id not in user_dict[int(did)].keys(): 
                    user_dict[int(did)][int(pure_user_id)] = pure_items_in_seq
                else:
                    #如果用户的序列长度小于当前的序列长度，则更新
                    if len(pure_items_in_seq) > len(user_dict[int(did)][int(pure_user_id)]):
                        print('pure_user_id: ', pure_user_id, 'updated seq: ', user_dict[int(did)][int(pure_user_id)], 'to', pure_items_in_seq)
                        user_dict[int(did)][int(pure_user_id)] = pure_items_in_seq
                user_set[int(did)].add(int(pure_user_id))
        print("total user number: ", sum(len(_) for _ in user_set))
        filter_user_id_list = []
        with open(os.path.join(args.input_path, short_name, f'{short_name}.filtered_user_id'), 'w',
                  encoding='utf-8') as file:
            for did in range(len(dataset_names)):
                print(f'Strict Mode: Writing [{dataset_names[did]}] user indexes down')
                filter_user_id = np.array(sorted(list(user_set[did])))
                print(len(user_set[did]), 'users in dataset', dataset_names[did])
                filter_user_id_list.append(filter_user_id)
                for uid in filter_user_id.tolist():
                    file.write(f'{did}-{uid}\n')
    feat_user_list = []
    for did, ds in enumerate(dataset_names):
        feat_path = os.path.join(args.input_path, short_name, f'{ds}.{args.suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, args.plm_size)
        print(f'Load {loaded_feat.shape} from {feat_path}.')
        if args.strict:
            for uid in filter_user_id_list[did].tolist():
                this_user_items_seq = np.array(user_dict[int(did)][uid], dtype=np.int64)
                this_user_emb = np.mean(loaded_feat[this_user_items_seq], axis=0)
                feat_user_list.append([this_user_emb])
    merged_feat = np.concatenate(feat_user_list, axis=0)
    print('Merged feature: ', merged_feat.shape)

    #上面把特征从suffix指定的文件中加载到merged_feat中
    save_index_path = os.path.join(
        args.output_path,
        short_name,
        f"{short_name}_user.OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x{args.n_centroid}{'.strict' if args.strict else ''}.index")

    if args.use_gpu:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024 * 1024 * 512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = args.subvector_num >= 56
    faiss.omp_set_num_threads(48)

    # 创建一个复合索引
    index = faiss.index_factory(args.plm_size,
                                f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x{args.n_centroid}",
                                faiss.METRIC_INNER_PRODUCT)
    index.verbose = True #为了输出详细信息
    if args.use_gpu:
        index = faiss.index_cpu_to_gpu(res, args.gpu_id, index, co)
    index.train(merged_feat)
    index.add(merged_feat)
    if args.use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, save_index_path)
