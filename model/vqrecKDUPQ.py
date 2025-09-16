import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
import numpy as np
import faiss
from recbole.model.abstract_recommender import SequentialRecommender


def log(t, eps = 1e-6):
    return torch.log(t + eps)


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)


def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)


def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)


class VQRecKDUPQ(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # VQRec args
        self.seq2bert = None
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.summary_mode = config['summary_mode']
        self.pq_codes = dataset.pq_codes
        self.id_dict = dataset.field2token_id['item_id']
        self.id_dict = self.reverse_dictionary(self.id_dict)
        self.text_emb = None
        self.uni_index = None
        self.code_cap = config['code_cap']
        self._codes_cache = {}
        self.pq_codes_user = dataset.pq_codes_user
        # self.neg_code = neg_code
        self.temperature = config['temperature']
        self.index_assignment_flag = False
        self.sinkhorn_iter = config['sinkhorn_iter']
        self.fake_idx_ratio = config['fake_idx_ratio']
        self.regularization_loss_weight = config['regularization_loss_weight']
        self.regularization_loss_weight_user = config['regularization_loss_weight_user']

        self.train_stage = config['train_stage']
        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'fedtrain'
        ], f'Unknown train stage: [{self.train_stage}]'

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.pq_code_embedding_share = nn.Embedding(
            self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)  # 加1是留给pad的
        self.pq_code_embedding_specific = nn.Embedding(
            self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
        self.reassigned_code_embedding = None

        #新加部分
        self.pq_code_user_embedding_share = nn.Embedding(
            self.code_dim * (self.code_cap), self.hidden_size)
        self.pq_code_user_embedding_specific = nn.Embedding(
            self.code_dim * (self.code_cap), self.hidden_size)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.trans_matrix = nn.Parameter(torch.randn(self.code_dim, self.code_cap + 1, self.code_cap + 1))

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            raise NotImplementedError()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self.apply(self._init_weights)
    def reverse_dictionary(self, d: dict):
        temp_dict = {}
        for k, v in d.items():
            if v == '[PAD]':
                continue
            try:
                temp_dict[v] = int(k[2:]) # int(k[2:])是因为k是'0-123'这种格式
            except ValueError:
                print(f"Warning: Key '{k}' with value '{v}' cannot be converted to int. Skipping.")
                continue
        return temp_dict

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def code_projection(self):
        doubly_stochastic_matrix = gumbel_sinkhorn(torch.exp(self.trans_matrix), n_iters=self.sinkhorn_iter)
        trans = differentiable_topk(doubly_stochastic_matrix.reshape(-1, self.code_cap + 1), 1)
        trans = torch.ceil(trans.reshape(-1, self.code_cap + 1, self.code_cap + 1))
        raw_embed = self.pq_code_embedding_specific.weight.reshape(self.code_dim, self.code_cap + 1, -1)
        trans_embed = torch.bmm(trans, raw_embed).reshape(-1, self.hidden_size)
        return trans_embed

    def get_summary(self, item_seq): 
        if self.summary_mode == "Mean":
            np_item_seq = np.array(item_seq.cpu().numpy(), dtype=np.int64)
            summary = self.text_emb[np_item_seq]
            summary = np.mean(summary, axis=-2)
            return summary
        elif self.summary_mode == "LLMEasy":
            return self.seq2bert.get_batch(item_seq, self.training)

    def get_codes(self, index, X_768: np.ndarray):
        """Encode style embeddings to PQ codes using an existing OPQ+IVF+PQ index.
        X_768: np.ndarray of shape [n, 768], dtype can be anything but will be cast to float32.
        Returns: np.ndarray of uint8 codes with shape [n, M].
        """
        # 1) Extract OPQ (if any) and the underlying IVF index (base class)
        if isinstance(index, faiss.IndexPreTransform):
            opq = faiss.downcast_VectorTransform(index.chain.at(0))  # OPQMatrix
            ivf_base = faiss.extract_index_ivf(index.index)           # IndexIVF (base)
        else:
            opq = None
            ivf_base = faiss.extract_index_ivf(index)                 # IndexIVF (base)

        # 2) Downcast to the concrete IVF subtype; prefer IVFPQ
        ivf_specific = ivf_base
        if hasattr(faiss, 'downcast_index'):
            try:
                ivf_specific = faiss.downcast_index(ivf_base)
            except Exception:
                ivf_specific = ivf_base

        if not isinstance(ivf_specific, faiss.IndexIVFPQ):
            raise TypeError(
                f"Underlying IVF is not IVFPQ (got {type(ivf_specific)}). Ensure the index was trained as IVFPQ (e.g., '...PQxx')."
            )

        ivfpq = ivf_specific  # now a proper IndexIVFPQ
        pq = ivfpq.pq  # ProductQuantizer

        # 3) Prepare input: float32 + contiguous
        X = X_768.astype('float32', copy=False)
        X = np.ascontiguousarray(X)  # [n, 768]

        # 4) Apply OPQ transform if present
        X_t = opq.apply_py(X) if opq is not None else X  # [n, d]

        # 5) Encode: IVF1 shortcut or general IVF*n*
        if ivfpq.nlist == 1:
            # list id always 0; prefer sa_encode if available
            try:
                codes = ivfpq.sa_encode(X_t)  # [n, M] uint8
            except Exception:
                centroids = faiss.vector_to_array(ivfpq.quantizer.xb).reshape(ivfpq.nlist, ivfpq.d)
                c0 = centroids[0]
                residual = X_t - c0
                codes = pq.compute_codes(residual)  # [n, M] uint8
        else:
            # General case: find nearest coarse centroid per vector
            try:
                _dists, labels = ivfpq.quantizer.search(X_t, 1)  # labels: [n,1]
                list_ids = labels[:, 0].astype(np.int32)
            except Exception:
                list_ids = np.empty((X_t.shape[0],), dtype=np.int32)
                ivfpq.quantizer.assign(X_t, 1, list_ids)

            try:
                codes = ivfpq.sa_encode(X_t)  # let faiss handle residual + PQ
            except Exception:
                centroids = faiss.vector_to_array(ivfpq.quantizer.xb).reshape(ivfpq.nlist, ivfpq.d)
                C = centroids[list_ids]
                residual = X_t - C
                codes = pq.compute_codes(residual)  # [n, M] uint8

        return codes
        
    def codes_remapp(self, codes):
        base_id = 0
        for i in range(codes.shape[1]): 
            codes[:, i] += base_id
            base_id += self.code_cap  # code_cap
        return torch.LongTensor(codes)
            
    def forward(self, item_seq, item_seq_len, user_id):
        if self.summary_mode == "Mean": 
            assert self.text_emb is not None, "Text embeddings must be loaded before forward pass"
        assert user_id is not None, "user_id must be provided for VQRecKDUPQ"
        if self.summary_mode != "Mean": 
            assert self.seq2bert is not None, "Seq2BertBank must be initialized for summary modes other than 'Mean'"
            self.seq2bert.id_dict = self.id_dict
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # ----- 用户风格SID的两种使用方式 -----
        # 动态：基于本条序列的summary即时编码（慢，但更细粒度）
        summary = self.get_summary(item_seq)  # np.ndarray [B, 768]

        # 可选：简单缓存（以 (user_id, seq_len) 作为key；对“前缀序列”数据集是唯一的）
        codes_tensor = None
        if self._codes_cache is not None:
            keys = [(int(u.item()), int(l.item())) for u, l in zip(user_id, item_seq_len)]
            miss_mask = []
            miss_idx = []
            cached_codes = []
            for i, k in enumerate(keys):
                if k in self._codes_cache:
                    cached_codes.append(self._codes_cache[k])
                    miss_mask.append(False)
                else:
                    cached_codes.append(None)
                    miss_mask.append(True)
                    miss_idx.append(i)
            if any(miss_mask):
                # 编码未命中部分（保持批量）
                miss_summary = summary[miss_idx]
                miss_codes = self.get_codes(self.uni_index, miss_summary)  # np.uint8 [m, M]
                # remap并转tensor
                miss_codes = self.codes_remapp(miss_codes).to(item_seq.device)
                # 写回缓存 & 组装
                mi = 0
                filled = []
                for i, k in enumerate(keys):
                    if miss_mask[i]:
                        self._codes_cache[k] = miss_codes[mi].detach().cpu()
                        filled.append(miss_codes[mi:mi+1])
                        mi += 1
                    else:
                        filled.append(cached_codes[i].to(item_seq.device).unsqueeze(0))
                codes_tensor = torch.cat(filled, dim=0)  # [B, M]
            else:
                codes_tensor = torch.stack([c.to(item_seq.device) for c in cached_codes], dim=0)
    
        # summary = self.get_summary(item_seq)
        # codes = self.get_codes(self.uni_index, summary)
        # codes = self.codes_remapp(codes)
        codes_emb = self.pq_code_user_embedding_specific(codes_tensor).mean(dim=-2)  # [B H]
        codes_emb = self.LayerNorm(codes_emb)
        codes_emb = self.dropout(codes_emb)
        #pq_code是(item_num, code_dim(32)), 其中每个dim的取值在[0, code_cap(256)]之间
        #若index为0，则表示padding
        pq_code_seq = self.pq_codes[item_seq]
        if self.index_assignment_flag:
            pq_code_emb = F.embedding(pq_code_seq, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pq_code_emb = self.pq_code_embedding_specific(pq_code_seq).mean(dim=-2)
        input_emb = pq_code_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        user_output = self.pq_code_user_embedding_specific(self.pq_codes_user[user_id]).mean(dim=-2)
        user_output = self.LayerNorm(user_output)
        user_output = self.dropout(user_output)
        output = self.concat_layer(torch.cat([output, codes_emb], dim=-1))
        #output = self.concat_layer(torch.cat([output, user_output], dim=-1))
        return output  # [B H],输出是个embedding

    def calculate_item_emb(self):
        if self.index_assignment_flag:
            pq_code_emb = F.embedding(self.pq_codes, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pq_code_emb = self.pq_code_embedding_specific(self.pq_codes).mean(dim=-2)  # self.pq_codes
        return pq_code_emb  # [B H]

    def generate_fake_neg_item_emb(self, item_index):
        rand_idx = torch.randint_like(input=item_index, high=self.code_cap)
        # flatten pq codes
        base_id = (torch.arange(self.code_dim).to(item_index.device) * (self.code_cap + 1)).unsqueeze(0)
        rand_idx = rand_idx + base_id + 1
        
        mask = torch.bernoulli(torch.full_like(item_index, self.fake_idx_ratio, dtype=torch.float))
        fake_item_idx = torch.where(mask > 0, rand_idx, item_index)
        return self.pq_code_embedding_specific(fake_item_idx).mean(dim=-2)

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_id = interaction['item_id']
        pos_pq_code = self.pq_codes[pos_id]
        if self.index_assignment_flag:
            pos_items_emb = F.embedding(pos_pq_code, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pos_items_emb = self.pq_code_embedding_specific(pos_pq_code).mean(dim=-2)
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature  # 乘转置后的得到2048x2048的，就是相当于每一条表示对每一个target的预测
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)  # 这行意思是为same_pos_id里面值为true时用的是[0]这个张量，是false时用的是neg_logits
        neg_logits = torch.exp(neg_logits).sum(dim=1).reshape(-1, 1)

        fake_item_emb = self.generate_fake_neg_item_emb(pos_pq_code)
        fake_item_emb = F.normalize(fake_item_emb, dim=-1)
        fake_logits = (seq_output * fake_item_emb).sum(dim=1, keepdim=True) / self.temperature
        fake_logits = torch.exp(fake_logits)

        loss = -torch.log(pos_logits / (neg_logits + fake_logits))
        return loss.mean()
    
    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))  # 这行是相当于把一个个单独的target拿出来和2048个target比较，在出现的位置标为true
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))  # 这行时生成主对角线全为true的元素
        # 如果一个target只在这一个batch里唯一，第一行和用torch.eye出来的元素应该完全一样，这两行的目的就是通过和主对角线元素异或的方式将重复出现的target标出来然后在做负对比的时候把这个给mask掉，因为这样的负样本对是不可用的
        return self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)

    def fedtrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)
        pos_id = interaction['item_id']
        pos_pq_code = self.pq_codes[pos_id]
        neg_items_emb = self.pq_code_embedding_specific(self.neg_code).mean(dim=-2)
        neg_items_emb = F.normalize(neg_items_emb, dim=1)
        pos_items_emb = self.pq_code_embedding_specific(pos_pq_code).mean(dim=-2)
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, neg_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.exp(neg_logits).sum(dim=1).reshape(-1, 1)

        fake_item_emb = self.generate_fake_neg_item_emb(pos_pq_code)
        fake_item_emb = F.normalize(fake_item_emb, dim=-1)
        fake_logits = (seq_output * fake_item_emb).sum(dim=1, keepdim=True) / self.temperature
        fake_logits = torch.exp(fake_logits)

        loss = -torch.log(pos_logits / (neg_logits + fake_logits))
        return loss.mean()

    def calculate_loss(self, interaction, global_embedding=None, global_embedding_user=None):
        

        # if self.train_stage == 'pretrain':
        #     return self.pretrain(interaction)
        # elif self.train_stage == 'fedtrain':
        #     return self.fedtrain(interaction)
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID] #😁新加了这一行
        user_id = user_id - 1
        #数据
        seq_output = self.forward(item_seq, item_seq_len, user_id)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            raise NotImplementedError()
        else:  # self.loss_type = 'CE'
            test_item_emb = self.calculate_item_emb()#所有的item的32位code求平均
            
            if self.temperature > 0:
                seq_output = F.normalize(seq_output, dim=-1)
                test_item_emb = F.normalize(test_item_emb, dim=-1)
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            
            if self.temperature > 0:
                logits /= self.temperature
            
            loss = self.loss_fct(logits, pos_items)
            #底下是mse kd设计，现在先屏蔽掉
            # if global_embedding is not None:
            #     mse = nn.MSELoss(reduction='mean')
            #     self.pq_code_embedding_share.load_state_dict(global_embedding.state_dict())
            #     regulaization_loss = mse(self.pq_code_embedding_specific.weight, self.pq_code_embedding_share.weight)
            #     loss += regulaization_loss * self.regularization_loss_weight
                
            # if global_embedding_user is not None: 
                
            #     mse = nn.MSELoss(reduction='mean')
            #     self.pq_code_user_embedding_share.load_state_dict(global_embedding_user.state_dict())
            #     regulaization_loss = mse(self.pq_code_user_embedding_specific.weight, self.pq_code_user_embedding_share.weight)
            #     loss += regulaization_loss * self.regularization_loss_weight_user

            return loss

    def predict(self, interaction):
        raise NotImplementedError()

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        user_id = user_id - 1  # Adjust for zero-based indexing
        seq_output = self.forward(item_seq, item_seq_len, user_id)
        test_items_emb = self.calculate_item_emb()
        
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_items_emb = F.normalize(test_items_emb, dim=-1)
        
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
