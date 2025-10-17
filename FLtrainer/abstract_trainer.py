from logging import getLogger
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import ensure_dir, \
    EvaluatorType, get_tensorboard, set_color, get_gpu_usage


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config_A, config_B, model_A, model_B):
        self.config_A = config_A
        self.config_B = config_B
        self.model_A = model_A
        self.model_B = model_B

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config_A, config_B, model_A, model_B):
        super(Trainer, self).__init__(config_A, config_B, model_A, model_B)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.learner = config_A['learner']
        self.learning_rate = config_A['learning_rate']
        self.epochs = config_A['epochs']
        self.eval_step = min(config_A['eval_step'], self.epochs)
        self.stopping_step = config_A['stopping_step']
        self.clip_grad_norm = config_A['clip_grad_norm']
        self.valid_metric = config_A['valid_metric'].lower()
        self.valid_metric_bigger = config_A['valid_metric_bigger']
        self.test_batch_size = config_A['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config_A['use_gpu']
        self.device = config_A['device']
        self.checkpoint_dir = config_A['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        self.weight_decay = config_A['weight_decay']
        self.config_A = config_A
        self.config_B = config_B

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict_A = dict()
        self.train_loss_dict_B = dict()
        self.optimizer_A = self._build_optimizer_A()
        self.optimizer_B = self._build_optimizer_B()

        self.eval_collector_A = Collector(config_A)
        self.evaluator_A = Evaluator(config_A)
        self.eval_collector_B = Collector(config_B) 
        self.evaluator_B = Evaluator(config_B)
        self.item_tensor = None
        self.tot_item_num = None

    def fit(self, train_data):
        pass

    def evaluate(self, eval_data):
        pass

    def _build_optimizer_A(self, **kwargs):
        params_A = kwargs.pop('params_A', self.model_A.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config_A['reg_weight'] and weight_decay and weight_decay * self.config_A['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params_A, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params_A, lr=learning_rate)
        return optimizer

    def _build_optimizer_B(self, **kwargs):
        params_B = kwargs.pop('params_B', self.model_B.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config_B['reg_weight'] and weight_decay and weight_decay * self.config_B['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params_B, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Bdam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Bdam optimizer')
            optimizer = optim.Adam(params_B, lr=learning_rate)
        return optimizer

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output_A(self, epoch_idx, s_time, e_time, losses):
        des = self.config_A['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss_A%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss_A', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _generate_train_loss_output_B(self, epoch_idx, s_time, e_time, losses):
        des = self.config_B['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss_B%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss_B', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)
    
    def _full_sort_batch_eval(self, batched_data, model):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, batched_data, model):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config['eval_type'] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config['eval_type'] == EvaluatorType.RANKING:
            col_idx = interaction[self.config['ITEM_ID_FIELD']]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i

    def evaluate(self, model, eval_data, eval_collector, evaluator, show_progress=False):
        if not eval_data:
            return
        
        model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config_A['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data, model)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        eval_collector.model_collect(model)
        struct = eval_collector.get_data_struct()
        result = evaluator.evaluate(struct)
        # self.wandblogger.log_eval_metrics(result, head='eval')
        return result

