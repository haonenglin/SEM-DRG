import os
import torch
from numpy import inf
import logging

class BaseTrainer(object):

    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        self.logger = logging.getLogger('modules.trainer')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            _handler = logging.StreamHandler()
            _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
            _handler.setFormatter(_fmt)
            self.logger.addHandler(_handler)
            log_dir = getattr(self.args, 'record_dir', os.path.join('.', 'records', 'TEST'))
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'trainer.log')
            _file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            _file_handler.setFormatter(_fmt)
            self.logger.addHandler(_file_handler)
            self.logger.propagate = False
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.report_interval = getattr(args, 'report_interval', 0)
        self.best_test_value = inf if self.mnt_mode == 'min' else -inf
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}, 'test': {self.mnt_metric_test: self.mnt_best}}

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("There's no GPU available on this machine,training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("The number of GPU's configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return (device, list_ids)

class DER_MOE_Trainer(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(DER_MOE_Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        self.logger.info(f'[{epoch}/{self.epochs}] Start to train in the training set.')
        report_every = getattr(self.args, 'log_interval', None)
        if report_every is None:
            tb = max(1, len(self.train_dataloader))
            report_every = max(1, tb // 10)
        for batch_idx, (images_id, images, reports_ids, reports_masks, batch_ent_lists) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = (images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device))
            device_ent_lists = []
            for feat_group in batch_ent_lists:
                if isinstance(feat_group, torch.Tensor):
                    device_ent_lists.append(feat_group.to(self.device))
                else:
                    device_ent_lists.append([f.to(self.device) if isinstance(f, torch.Tensor) else f for f in feat_group])
            output = self.model(images, reports_ids, mode='train', ent_lists=device_ent_lists)
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if batch_idx % report_every == 0:
                self.logger.info(f'[{epoch}/{self.epochs}] Step: {batch_idx}/{len(self.train_dataloader) - 1}, Training Loss: {loss.item():.5f}.')
            log = {'train_loss': train_loss / len(self.train_dataloader)}
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = ([], [])
            records = []
            self.logger.info(f'[{epoch}/{self.epochs}] Start to evaluate in the test set.')
            for batch_idx, (images_id, images, reports_ids, reports_masks, batch_ent_lists) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = (images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device))
                device_ent_lists = []
                for feat_group in batch_ent_lists:
                    if isinstance(feat_group, torch.Tensor):
                        device_ent_lists.append(feat_group.to(self.device))
                    else:
                        device_ent_lists.append([f.to(self.device) if isinstance(f, torch.Tensor) else f for f in feat_group])
                output = self.model(images, mode='sample', ent_lists=device_ent_lists)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                for i, sid in enumerate(images_id):
                    records.append({'id': str(sid), 'ground_truth': ground_truths[i], 'generated': reports[i]})
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)}, {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            self._last_test_records = records
        self.lr_scheduler.step()
        return log
