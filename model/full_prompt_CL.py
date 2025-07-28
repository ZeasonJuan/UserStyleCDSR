import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder,MLPLayers
from recbole.model.abstract_recommender import SequentialRecommender


class id_prompt(SequentialRecommender):
    def __init__(self, config, dataset, vqrecCL):
        super().__init__(config, dataset)
        self.pq_codes = dataset.pq_codes
        self.index_assignment_flag = False
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']
        self.reassigned_code_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.prompts = nn.Parameter(torch.randn(1024, self.hidden_size))
        nn.init.xavier_uniform_(self.prompts)
        self.item_trm_encoder = TransformerEncoder(
            n_layers=2,
            n_heads=2,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.vqrecCL = vqrecCL
        self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)
        self.concat_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
        # self.concat_layer = MLPLayers(layers=[3 * self.hidden_size, self.hidden_size], activation='leakyrelu')
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        prompt = self.get_prompt(seq_out=feature_output.unsqueeze(1)).squeeze(1)
        output_concat = torch.cat((prompt, item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)#一个embedding
        return seq_output

    def get_prompt(self, seq_out):
        prompt = self.prompts.unsqueeze(1)
        prompt, _ = self.attn_layer(seq_out, prompt, prompt)
        return prompt

    def calculate_item_emb(self):
        pq_code_emb = self.vqrecCL.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores

class id_prompt_no_prompt(SequentialRecommender):
    def __init__(self, config, dataset, vqrecCL):
        super().__init__(config, dataset)
        self.pq_codes = dataset.pq_codes
        self.index_assignment_flag = False
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']
        self.reassigned_code_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.item_trm_encoder = TransformerEncoder(
            n_layers=2,
            n_heads=2,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.vqrecCL = vqrecCL
        self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.concat_layer = MLPLayers(layers=[3 * self.hidden_size, self.hidden_size], activation='leakyrelu')
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)#一个embedding
        return seq_output

    def calculate_item_emb(self):
        pq_code_emb = self.vqrecCL.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores



class id_prompt_CL(SequentialRecommender):#这个是原始的ptune设计，在融合后的seq基础上进行si、ss
    def __init__(self, config, dataset, vqrecCL):
        super().__init__(config, dataset)
        self.pq_codes = dataset.pq_codes
        self.index_assignment_flag = False
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']
        self.reassigned_code_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.prompts = nn.Parameter(torch.randn(1024, self.hidden_size))
        nn.init.xavier_uniform_(self.prompts)
        self.item_trm_encoder = TransformerEncoder(
            n_layers=2,
            n_heads=2,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.vqrecCL = vqrecCL
        self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)
        self.concat_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
        # self.concat_layer = MLPLayers(layers=[3 * self.hidden_size, self.hidden_size], activation='leakyrelu')
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        prompt = self.get_prompt(seq_out=feature_output.unsqueeze(1)).squeeze(1)
        output_concat = torch.cat((prompt, item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)#一个embedding
        return seq_output

    def get_prompt(self, seq_out):
        prompt = self.prompts.unsqueeze(1)
        prompt, _ = self.attn_layer(seq_out, prompt, prompt)
        return prompt

    def calculate_item_emb(self):
        pq_code_emb = self.vqrecCL.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb
    
    def contrastive_loss(self, seq_output, pos_items_emb, temperature=0.1):
        """
        seq_output      : [B, H]  – 序列向量 s
        pos_items_emb   : [B, H]  – 对应正样本 v
        返回两个对比损失之和：
        (1) s 与 augment(s) / 其他 s
        (2) s 与 v / 其他 v
        """
        
        # -------- 预处理：L2 norm ----------
        seq_output = F.normalize(seq_output, dim=-1)
        pos_items_emb = F.normalize(pos_items_emb, dim=-1)


        B, H = seq_output.shape

        # -------- (1) s ↔ s_aug 对比 --------
        # 简单做法：用 Dropout 创建 s_aug
        s_aug = F.dropout(seq_output, p=0.05, training=True)
        s_aug = F.normalize(s_aug, dim=-1)

        logits_ss = torch.matmul(seq_output, s_aug.t()) / temperature          # [B,B]
        labels = torch.arange(B, device=seq_output.device)
        loss_ss = F.cross_entropy(logits_ss, labels)

        # -------- (2) s ↔ v 对比 --------
        logits_sv = torch.matmul(seq_output, pos_items_emb.t()) / temperature  # [B,B]
        loss_sv = F.cross_entropy(logits_sv, labels)

        return loss_ss * self.vqrecCL.contrastive_loss_weight_ss + loss_sv * self.vqrecCL.contrastive_loss_weight_sv

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_code = self.pq_codes[pos_items]
        pos_items_emb = self.vqrecCL.pq_code_embedding(pos_items_code).mean(dim=-2)
        contrastive_loss = self.contrastive_loss(seq_output, pos_items_emb, self.vqrecCL.contrastive_temperature)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss + contrastive_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores
    

class id_prompt_CL_SID_id(SequentialRecommender):# SIDseq和iDseq对比 ➕ SIDseq和PosID对比
    def __init__(self, config, dataset, vqrecCL):
        super().__init__(config, dataset)
        self.pq_codes = dataset.pq_codes
        self.index_assignment_flag = False
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']
        self.reassigned_code_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.prompts = nn.Parameter(torch.randn(1024, self.hidden_size))
        nn.init.xavier_uniform_(self.prompts)
        self.item_trm_encoder = TransformerEncoder(
            n_layers=2,
            n_heads=2,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.vqrecCL = vqrecCL
        self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)
        self.concat_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
        # self.concat_layer = MLPLayers(layers=[3 * self.hidden_size, self.hidden_size], activation='leakyrelu')
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        prompt = self.get_prompt(seq_out=feature_output.unsqueeze(1)).squeeze(1)
        output_concat = torch.cat((prompt, item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)#一个embedding
        return seq_output
    
    def half_forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        prompt = self.get_prompt(seq_out=feature_output.unsqueeze(1)).squeeze(1)
        return prompt, item_output, feature_output

    def get_prompt(self, seq_out):
        prompt = self.prompts.unsqueeze(1)
        prompt, _ = self.attn_layer(seq_out, prompt, prompt)
        return prompt

    def calculate_item_emb(self):
        pq_code_emb = self.vqrecCL.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb
    
    def contrastive_loss(self, seq_output, pos_items_emb, id_seq_embedding, temperature=0.1):
        """
        seq_output      : [B, H]  – 序列向量 s
        pos_items_emb   : [B, H]  – 对应正样本 v
        返回两个对比损失之和：
        (1) s 与 augment(s) / 其他 s
        (2) s 与 v / 其他 v
        """
        
        # -------- 预处理：L2 norm ----------
        seq_output = F.normalize(seq_output, dim=-1)
        pos_items_emb = F.normalize(pos_items_emb, dim=-1)
        id_seq_embedding = F.normalize(id_seq_embedding, dim=-1)


        B, H = seq_output.shape

        # -------- (1) s ↔ id_s 对比 --------


        logits_si = torch.matmul(seq_output, id_seq_embedding.t()) / temperature          # [B,B]
        labels = torch.arange(B, device=seq_output.device)
        loss_ss = F.cross_entropy(logits_si, labels)

        # -------- (2) s ↔ v 对比 --------
        logits_sv = torch.matmul(seq_output, pos_items_emb.t()) / temperature  # [B,B]
        loss_sv = F.cross_entropy(logits_sv, labels)

        return loss_ss * self.vqrecCL.contrastive_loss_weight_ss + loss_sv * self.vqrecCL.contrastive_loss_weight_sv

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        #后面一半forward
        prompt, item_output, feature_output = self.half_forward(item_seq, item_seq_len)
        output_concat = torch.cat((prompt, item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)



        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_code = self.pq_codes[pos_items]
        pos_items_emb = self.vqrecCL.pq_code_embedding(pos_items_code).mean(dim=-2)


        contrastive_loss = self.contrastive_loss(feature_output, pos_items_emb, item_output, self.vqrecCL.contrastive_temperature)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss + contrastive_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores

class id_prompt_CL_SID_id_no_prompt(SequentialRecommender):# SIDseq和iDseq对比 ➕ SIDseq和PosID对比
    def __init__(self, config, dataset, vqrecCL):
        super().__init__(config, dataset)
        self.pq_codes = dataset.pq_codes
        self.index_assignment_flag = False
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']
        self.reassigned_code_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        
        self.item_trm_encoder = TransformerEncoder(
            n_layers=2,
            n_heads=2,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.vqrecCL = vqrecCL
        self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.concat_layer = MLPLayers(layers=[3 * self.hidden_size, self.hidden_size], activation='leakyrelu')
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)#一个embedding
        return seq_output
    
    def half_forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        return item_output, feature_output


    def calculate_item_emb(self):
        pq_code_emb = self.vqrecCL.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb
    
    def contrastive_loss(self, seq_output, pos_items_emb, id_seq_embedding, temperature=0.1):
    
        
        # -------- 预处理：L2 norm ----------
        seq_output = F.normalize(seq_output, dim=-1)
        pos_items_emb = F.normalize(pos_items_emb, dim=-1)
        id_seq_embedding = F.normalize(id_seq_embedding, dim=-1)


        B, H = seq_output.shape

        # -------- (1) s ↔ id_s 对比 --------


        logits_si = torch.matmul(seq_output, id_seq_embedding.t()) / temperature          # [B,B]
        labels = torch.arange(B, device=seq_output.device)
        loss_ss = F.cross_entropy(logits_si, labels)

        # -------- (2) s ↔ v 对比 --------
        logits_sv = torch.matmul(seq_output, pos_items_emb.t()) / temperature  # [B,B]
        loss_sv = F.cross_entropy(logits_sv, labels)

        return loss_ss * self.vqrecCL.contrastive_loss_weight_ss + loss_sv * self.vqrecCL.contrastive_loss_weight_sv

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        #后面一半forward
        item_output, feature_output = self.half_forward(item_seq, item_seq_len)
        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)



        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_code = self.pq_codes[pos_items]
        pos_items_emb = self.vqrecCL.pq_code_embedding(pos_items_code).mean(dim=-2)


        contrastive_loss = self.contrastive_loss(feature_output, pos_items_emb, item_output, self.vqrecCL.contrastive_temperature)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss + contrastive_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores


class id_prompt_CL_no_prompt(SequentialRecommender):#这是去了prompt之后，在融合后的seq基础上进行si、ss
    def __init__(self, config, dataset, vqrecCL):
        super().__init__(config, dataset)
        self.pq_codes = dataset.pq_codes
        self.index_assignment_flag = False
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']
        self.reassigned_code_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        
        self.item_trm_encoder = TransformerEncoder(
            n_layers=2,
            n_heads=2,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.vqrecCL = vqrecCL
        self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.concat_layer = MLPLayers(layers=[3 * self.hidden_size, self.hidden_size], activation='leakyrelu')
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrecCL.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrecCL.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrecCL.forward(item_seq, item_seq_len)
        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)#一个embedding
        return seq_output


    def calculate_item_emb(self):
        pq_code_emb = self.vqrecCL.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb
    
    def contrastive_loss(self, seq_output, pos_items_emb, temperature=0.1):
        """
        seq_output      : [B, H]  – 序列向量 s
        pos_items_emb   : [B, H]  – 对应正样本 v
        返回两个对比损失之和：
        (1) s 与 augment(s) / 其他 s
        (2) s 与 v / 其他 v
        """
        
        # -------- 预处理：L2 norm ----------
        seq_output = F.normalize(seq_output, dim=-1)
        pos_items_emb = F.normalize(pos_items_emb, dim=-1)


        B, H = seq_output.shape

        # -------- (1) s ↔ s_aug 对比 --------
        # 简单做法：用 Dropout 创建 s_aug
        s_aug = F.dropout(seq_output, p=0.05, training=True)
        s_aug = F.normalize(s_aug, dim=-1)

        logits_ss = torch.matmul(seq_output, s_aug.t()) / temperature          # [B,B]
        labels = torch.arange(B, device=seq_output.device)
        loss_ss = F.cross_entropy(logits_ss, labels)

        # -------- (2) s ↔ v 对比 --------
        logits_sv = torch.matmul(seq_output, pos_items_emb.t()) / temperature  # [B,B]
        loss_sv = F.cross_entropy(logits_sv, labels)

        return loss_ss * self.vqrecCL.contrastive_loss_weight_ss + loss_sv * self.vqrecCL.contrastive_loss_weight_sv

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_code = self.pq_codes[pos_items]
        pos_items_emb = self.vqrecCL.pq_code_embedding(pos_items_code).mean(dim=-2)
        contrastive_loss = self.contrastive_loss(seq_output, pos_items_emb, self.vqrecCL.contrastive_temperature)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss + contrastive_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores

