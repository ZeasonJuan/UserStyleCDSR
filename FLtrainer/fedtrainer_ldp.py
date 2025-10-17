import os
from logging import getLogger
from time import time
from datetime import date
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from torch.nn import functional as F
from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage, WandbLogger

from .abstract_trainer import Trainer


class FedtrainTrainer(Trainer):
    r"""PretrainTrainer is designed for pre-training.
    It can be inherited by the trainer which needs pre-training and fine-tuning.
    """

    def __init__(self, config_A, config_B, model_A, model_B, global_embedding):
        super(FedtrainTrainer, self).__init__(config_A, config_B, model_A, model_B)
        self.pretrain_epochs = self.config_A['pretrain_epochs']
        self.device = self.config_A['device']
        self.num_clients = 2
        self.save_step = self.config_A['save_step']
        self.global_embedding = global_embedding
        self.epsilon = self.config_A['epsilon']  # 隐私预算，控制噪声强度
        self.delta = 1e-5  # 失败概率，控制隐私保证的可靠性
        self.num_bits = self.config_A['num_bits']  # 量化的位数，控制梯度的精度[6, 10]
        self.num_buckets = 2 ** self.num_bits  # 量化的桶数，等于 2 的位数次方
        self.clip_threshold = 0.12  # 梯度的裁剪阈值，控制梯度的范围
        self.scale_factor = self.clip_threshold / self.num_buckets  # 缩放因子，用于将梯度转换为整数
        self.prob_p = (np.exp(self.epsilon) + 1) / (np.exp(self.epsilon) + 2)  # 随机响应中翻转概率的分子
        self.prob_q = 1 / (np.exp(self.epsilon) + 2)  # 随机响应中翻转概率的分母
        
        # self.attn_layer = nn.MultiheadAttention(config_A['hidden_size'], num_heads=4).to(config_A['device'])

    # 定义一个函数，用于对梯度进行量化，即将梯度转换为整数值
    def quantize(self, gradient):
        # 对梯度进行裁剪，使其在 [-clip_threshold, clip_threshold] 范围内
        gradient = torch.clamp(gradient, -self.clip_threshold, self.clip_threshold)
        # 对梯度进行缩放和四舍五入，使其在 [0, num_buckets - 1] 范围内
        gradient = torch.round((gradient + self.clip_threshold) / self.scale_factor)
        # 将浮点型张量转换为整型张量
        gradient = gradient.type(torch.int64)
        return gradient
    
    def _train_epoch_A(self, train_data, epoch_idx, global_embedding, loss_func=None, show_progress=False):
        # for param in self.model_B.parameters():
        #     param.requires_grad=False
        self.model_A.train()
        embedding_grad_sum = torch.zeros_like(self.model_A.pq_code_embedding.weight)
        loss_func = loss_func or self.model_A.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train_A {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer_A.zero_grad()
            losses = self.model_A.calculate_loss(interaction) #+ 0.005 * torch.norm(self.model_A.pq_code_embedding.weight - global_embedding.weight) ** 2
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model_A.parameters(), **self.clip_grad_norm)
            self.optimizer_A.step()
            embedding_grad_sum += self.model_A.pq_code_embedding.weight.grad.clone()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('Loss: ' + str(losses.item()), 'yellow'))
        return total_loss, embedding_grad_sum

    def _train_epoch_B(self, train_data, epoch_idx, global_embedding, loss_func=None, show_progress=False):
        self.model_B.train()
        embedding_grad_sum = torch.zeros_like(self.model_B.pq_code_embedding.weight)
        loss_func = loss_func or self.model_B.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,#TrainDataLoader的实例
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train_B {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer_B.zero_grad()
            losses = self.model_B.calculate_loss(interaction) #+ 0.005 * torch.norm(self.model_B.pq_code_embedding.weight - global_embedding.weight) ** 2
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model_B.parameters(), **self.clip_grad_norm)
            self.optimizer_B.step()
            embedding_grad_sum += self.model_B.pq_code_embedding.weight.grad.clone()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('Loss: ' + str(losses.item()), 'yellow'))
        return total_loss, embedding_grad_sum

    # 定义一个函数，用于对量化后的梯度进行随机响应，即以一定的概率对其进行翻转或置换
    def randomize(self, quantized_gradient):
        # 获取张量的形状和元素个数
        shape = quantized_gradient.shape
        num_elements = quantized_gradient.numel()
        # 将张量展平为一维向量
        quantized_gradient = quantized_gradient.flatten()
        # 创建一个与梯度形状相同的随机向量，每个元素服从伯努利分布，参数为 prob_p
        coin_flips = torch.bernoulli(torch.tensor([self.prob_p] * num_elements, device=self.device))
        # 创建一个与梯度形状相同的随机向量，每个元素服从均匀分布，范围为 [0, num_buckets - 1]
        noise = torch.randint(0, self.num_buckets, (num_elements,), device=self.device)
        # 根据随机向量对梯度进行翻转或置换
        randomized_gradient = (quantized_gradient + coin_flips * noise) % self.num_buckets  # 置换操作，等价于加上一个随机偏移量并取模
        # 将一维向量恢复为原始张量的形状
        randomized_gradient = randomized_gradient.reshape(shape)
        return randomized_gradient

    # 定义一个函数，用于对随机响应后的梯度进行解码和校正，以消除噪声的影响，并进行聚合更新
    def aggregate(self, randomized_gradients, weights):
        # 获取张量的形状和元素个数
        shape = randomized_gradients[0].shape
        num_elements = randomized_gradients[0].numel()
        # 将每个客户端的张量展平为一维向量，并拼接在一起，形成一个二维矩阵
        randomized_gradients = torch.stack(
            [randomized_gradient.flatten() for randomized_gradient in randomized_gradients])
        # 创建一个与矩阵形状相同的常数矩阵，每个元素为 prob_p - prob_q
        prob_diffs = torch.tensor([(self.prob_p - self.prob_q)] * num_elements * self.num_clients, device=self.device).reshape(self.num_clients, num_elements)
        # 对矩阵的每一列（即每个参数）进行解码和校正，得到去噪后的梯度向量
        # decoded_gradients = (randomized_gradients * prob_diffs).sum(0) / (self.prob_p - self.prob_q) / self.num_clients
        decoded_gradients = (randomized_gradients * prob_diffs * weights.unsqueeze(1)).sum(0) / (self.prob_p - self.prob_q)
        # 将去噪后的梯度向量还原为原始梯度的形状，并进行反缩放，得到聚合后的梯度张量
        aggregated_gradient = (decoded_gradients.reshape(shape) - self.clip_threshold) * self.scale_factor
        return aggregated_gradient

    def save_pretrained_model_A(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config_A,
            'epoch': epoch,
            'state_dict': self.model_A.state_dict(),
            'optimizer': self.optimizer_A.state_dict(),
            'other_parameter': self.model_A.other_parameter(),
        }
        torch.save(state, saved_model_file)

    def save_pretrained_model_B(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config_B,
            'epoch': epoch,
            'state_dict': self.model_B.state_dict(),
            'optimizer': self.optimizer_B.state_dict(),
            'other_parameter': self.model_B.other_parameter(),
        }
        torch.save(state, saved_model_file)

    def fedtrain(self, train_data_A, train_data_B, test_data_A, test_data_B, weight, verbose=True, show_progress=False):
        
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            client_gradients = []
            training_start_time = time()
            train_loss_A, embedding_grad_A = self._train_epoch_A(train_data_A, epoch_idx, self.global_embedding, show_progress=show_progress)
            train_loss_B, embedding_grad_B = self._train_epoch_B(train_data_B, epoch_idx, self.global_embedding, show_progress=show_progress)
            if (epoch_idx + 1) % 2 == 0:
                test_result_A = self.evaluate(self.model_A, test_data_A, self.eval_collector_A, self.evaluator_A, show_progress=show_progress)
                test_result_B = self.evaluate(self.model_B, test_data_B, self.eval_collector_B, self.evaluator_B, show_progress=show_progress)
                self.logger.info(f'Epoch {epoch_idx}: A test result {test_result_A}')
                self.logger.info(f'Epoch {epoch_idx}: B test result {test_result_B}')

            self.train_loss_dict_A[epoch_idx] = sum(train_loss_A) if isinstance(train_loss_A, tuple) else train_loss_A
            self.train_loss_dict_B[epoch_idx] = sum(train_loss_B) if isinstance(train_loss_B, tuple) else train_loss_B
            training_end_time = time()
            train_loss_output_A = \
                self._generate_train_loss_output_A(epoch_idx, training_start_time, training_end_time, train_loss_A)
            train_loss_output_B = \
                self._generate_train_loss_output_B(epoch_idx, training_start_time, training_end_time, train_loss_B)
            if verbose:
                self.logger.info(train_loss_output_A)
                self.logger.info(train_loss_output_B)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss_A)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss_B)

            # 聚合embedding
            #将梯度变成整型，为了差分隐私保护做准备
            client_gradients_A = self.quantize(embedding_grad_A.clone())
            client_gradients_A = self.randomize(client_gradients_A)
            client_gradients_B = self.quantize(embedding_grad_B.clone())
            client_gradients_B = self.randomize(client_gradients_B)
            client_gradients.append(client_gradients_A)
            client_gradients.append(client_gradients_B)
            global_embedding_params = self.global_embedding.state_dict()
            for key in global_embedding_params.keys():
                global_embedding_params[key] = global_embedding_params[key] - \
                                               self.learning_rate * self.aggregate(client_gradients, weight)
            self.global_embedding.load_state_dict(global_embedding_params)
            self.model_A.pq_code_embedding.weight.data = self.global_embedding.weight.data.clone()
            self.model_B.pq_code_embedding.weight.data = self.global_embedding.weight.data.clone()

            if (epoch_idx + 1) % (self.save_step) == 0:
                saved_model_file_A = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}-{}.pth'.format(self.config_A['model'], self.config_A['dataset'], str(epoch_idx + 1), str(date.today()))
                )
                saved_model_file_B = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}-{}.pth'.format(self.config_B['model'], self.config_B['dataset'], str(epoch_idx + 1), str(date.today()))
                )
                self.save_pretrained_model_A(epoch_idx, saved_model_file_A)
                self.save_pretrained_model_B(epoch_idx, saved_model_file_B)
                update_output_A = set_color('Saving current', 'blue') + ': %s' % saved_model_file_A
                update_output_B = set_color('Saving current', 'blue') + ': %s' % saved_model_file_B
                if verbose:
                    self.logger.info(update_output_A)
                    self.logger.info(update_output_B)

        return self.best_valid_score, self.best_valid_result