import csv
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from .trainer import DER_MOE_Trainer

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

def atomic_checkpoint(path, state):
    tmp = Path(str(path) + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)

def check_reports(records, examples):
    expected = {str(r['id']): r['report'] for r in examples}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError('Report count differs from test sample count')
    seen = set()
    for r in records:
        if set(r) != {'id', 'ground_truth', 'generated'} or not all((isinstance(v, str) for v in r.values())):
            raise ValueError('Report does not match Sample.json schema')
        if r['id'] in seen or r['id'] not in expected or r['ground_truth'] != expected[r['id']]:
            raise ValueError('Duplicate/wrong test ID or ground truth mismatch')
        seen.add(r['id'])

class SmokeLoader:

    def __init__(self, loader, count, pad_length=None):
        self.loader, self.count, self.pad_length = (loader, count, pad_length)
        self.dataset, self.generator = (loader.dataset, loader.generator)

    def __len__(self):
        return min(len(self.loader), self.count)

    def __iter__(self):
        for batch in itertools.islice(self.loader, self.count):
            if self.pad_length:
                batch = list(batch)
                pad = self.pad_length - batch[2].shape[1]
                batch[2] = torch.nn.functional.pad(batch[2], (0, pad))
                batch[3] = torch.nn.functional.pad(batch[3], (0, pad))
            yield batch

def rng_state(train, test):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), train_loader=train.generator.get_state(), test_loader=test.generator.get_state())

def restore_rng(state, train, test):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    train.generator.set_state(state['train_loader'])
    test.generator.set_state(state['test_loader'])

class FoldTrainer(DER_MOE_Trainer):

    def __init__(self, model, criterion, metrics, optimizer, args, scheduler, train, test, provenance, report_examples=None):
        logger = logging.getLogger('modules.trainer')
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        for handler in (logging.StreamHandler(), logging.FileHandler(Path(args.save_dir) / 'trainer.log', mode='a')):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        logger.propagate = False
        resume = args.resume
        args.resume = None
        super().__init__(model, criterion, metrics, optimizer, args, scheduler, train, None, test)
        args.resume = resume
        self.provenance = provenance
        self.not_improved_count = 0
        self.best_log = None
        self.best_reports = None
        self.training_finished = False
        self.report_examples = test.dataset.examples if report_examples is None else report_examples
        self.raw_test = {str(r['id']): r['report'] for r in self.report_examples}
        if resume:
            self.resume_fold(resume)

    def resume_fold(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if ck.get('run_identity') != self.provenance['identity']:
            raise ValueError('Resume rejected: checkpoint belongs to another fold/seed/config/code or is historical')
        self.model.load_state_dict(ck['state_dict'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.lr_scheduler.load_state_dict(ck['scheduler'])
        self.start_epoch = ck['epoch'] + 1
        self.mnt_best = ck['monitor_best']
        self.not_improved_count = ck['not_improved_count']
        self.best_log, self.best_reports = (ck['best_log'], ck['best_reports'])
        self.training_finished = ck['training_finished']
        restore_rng(ck['rng'], self.train_dataloader, self.test_dataloader)
        best_path = Path(self.args.save_dir) / 'model_best.pth'
        if self.best_log['epoch'] == ck['epoch']:
            shutil.copyfile(path, str(best_path) + '.tmp')
            os.replace(str(best_path) + '.tmp', best_path)
        else:
            best = torch.load(best_path, map_location='cpu', mmap=True, weights_only=False)
            if best['epoch'] != self.best_log['epoch'] or best.get('run_identity') != ck['run_identity']:
                raise ValueError('Best checkpoint is inconsistent with current checkpoint')
            del best
        write_json(Path(self.args.save_dir) / 'best_test_reports.json', self.best_reports)
        log_path = Path(self.args.save_dir) / 'df_epoch_logs.jsonl'
        lines = log_path.read_text().splitlines()
        logs = []
        for index, line in enumerate(lines):
            try:
                logs.append(json.loads(line))
            except ValueError:
                if index != len(lines) - 1:
                    raise
        logs = [row for row in logs if row['epoch'] <= ck['epoch']]
        if not logs or logs[-1]['epoch'] != ck['epoch']:
            raise ValueError('Committed checkpoint epoch is missing from metric log')
        log_path.write_text(''.join((json.dumps(row) + '\n' for row in logs)))
        for p in Path(self.args.save_dir).glob('epoch_*_test_reports.json'):
            if int(p.name.split('_')[1]) > ck['epoch']:
                p.unlink()
        self.logger.info('Resumed epoch %d, optimizer/scheduler/RNG/early-stop state restored', ck['epoch'])

    def save_fold(self, epoch, improved):
        state = dict(epoch=epoch, state_dict=self.model.state_dict(), optimizer=self.optimizer.state_dict(), monitor_best=self.mnt_best, scheduler=self.lr_scheduler.state_dict(), rng=rng_state(self.train_dataloader, self.test_dataloader), not_improved_count=self.not_improved_count, best_log=self.best_log, best_reports=self.best_reports, training_finished=self.training_finished, run_identity=self.provenance['identity'])
        out = Path(self.args.save_dir)
        atomic_checkpoint(out / 'current_checkpoint.pth', state)
        if improved:
            shutil.copyfile(out / 'current_checkpoint.pth', out / 'model_best.pth.tmp')
            os.replace(out / 'model_best.pth.tmp', out / 'model_best.pth')
            write_json(out / 'best_test_reports.json', self.best_reports)
        self.logger.info('Saving checkpoint: current_checkpoint.pth; best=%s', improved)

    def train(self):
        self.logger.info('Training Started: fold=%d seed=%d; test selects best/early stopping; val unused', self.args.fold, self.args.seed)
        self.logger.info('Configuration: %s', json.dumps(self.provenance['config'], ensure_ascii=False))
        end = 1 if self.args.smoke else self.epochs
        for epoch in range(self.start_epoch, end + 1):
            if self.training_finished:
                break
            log = {'epoch': epoch, **self._train_epoch(epoch)}
            value = log['test_' + self.args.monitor_metric]
            if not math.isfinite(value):
                raise ValueError('Non-finite checkpoint selection metric')
            for record in self._last_test_records:
                record['ground_truth'] = self.raw_test[record['id']]
            examples = self.report_examples
            if self.args.smoke:
                ids = {r['id'] for r in self._last_test_records}
                examples = [r for r in examples if str(r['id']) in ids]
            check_reports(self._last_test_records, examples)
            improved = value >= self.mnt_best
            if improved:
                self.mnt_best, self.best_log = (value, log.copy())
                self.best_reports = list(self._last_test_records)
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
            self.training_finished = epoch == end or self.not_improved_count > self.early_stop
            for key, val in log.items():
                self.logger.info('\t%-15s: %s', key, val)
            with (Path(self.args.save_dir) / 'df_epoch_logs.jsonl').open('a') as f:
                f.write(json.dumps(log) + '\n')
                f.flush()
                os.fsync(f.fileno())
            if self.report_interval and epoch % self.report_interval == 0:
                write_json(Path(self.args.save_dir) / f'epoch_{epoch}_test_reports.json', self._last_test_records)
            self.save_fold(epoch, improved)
        if self.best_log is None:
            raise RuntimeError('No completed training epoch')
        write_json(Path(self.args.save_dir) / 'Sample.json', self.best_reports)
        row = dict(self.best_log, time=time.asctime(), seed=self.args.seed, best_model_from='test')
        with (Path(self.args.save_dir) / 'df.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        self.logger.info('Best results (w.r.t %s) in test set: %s', self.args.monitor_metric, self.best_log)
        for handler in self.logger.handlers:
            handler.flush()

def run_fold(args):
    import fcntl
    import sys
    from .tokenizers import Tokenizer
    from DF_Dataloader.dataloaders import DACGdataLoader
    from models.sem_drg import SEMDRG
    from .loss import compute_loss
    from .metrics import compute_scores
    from .optimizers import build_optimizer, build_lr_scheduler

    root = Path(__file__).resolve().parents[1]
    if os.environ.get('PYTHONHASHSEED') != str(args.seed) or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        env = dict(os.environ, PYTHONHASHSEED=str(args.seed), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        os.execve(sys.executable, [sys.executable, str(root / 'train.py'), *sys.argv[1:]], env)
    if not torch.cuda.is_available():
        raise RuntimeError('SEM-DRG requires a CUDA GPU')
    if shutil.which('java') is None:
        raise RuntimeError('Install Java to compute METEOR')
    data = json.loads(Path(args.ann_path).read_text())
    all_ids = []
    for split in ('train', 'val', 'test'):
        if split not in data or not isinstance(data[split], list):
            raise ValueError(f'Missing annotation split: {split}')
        for row in data[split]:
            all_ids.append(str(row['id']))
            if not isinstance(row['report'], str) or not row['image_path']:
                raise ValueError('Each sample needs a report and image_path')
            path = Path(args.image_dir) / row['image_path'][0]
            if not path.is_file():
                raise FileNotFoundError(path)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Sample IDs must be unique across splits')
    if not data['train'] or not data['test']:
        raise ValueError('Train and test splits must be nonempty')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    tokenizer = Tokenizer(args)
    config = {k: v for k, v in vars(args).items() if k not in ('resume', 'print_config')}
    assets = sorted(Path(args.entity_dir).glob('*.npy'))
    if len(assets) != 9:
        raise ValueError('Expected nine entity feature arrays')
    sources = sorted(root.rglob('*.py'))
    provenance = dict(config=config,
        annotation_sha256=hashlib.sha256(Path(args.ann_path).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        asset_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets},
        vocabulary_sha256=digest(tokenizer.token2idx))
    provenance['identity'] = digest(provenance)
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = out / 'current_checkpoint.pth'
        previous = out / 'run_manifest.json'
        if previous.exists():
            if json.loads(previous.read_text())['identity'] != provenance['identity']:
                raise ValueError('Run configuration, source, or data changed; use a new --run name')
            if args.resume != 'auto':
                raise ValueError('Run already exists; use --resume auto or a new --run name')
        args.resume = str(current) if args.resume == 'auto' and current.exists() else None
        write_json(previous, provenance)
        write_json(out / 'vocabulary.json', tokenizer.token2idx)
        train = DACGdataLoader(args, tokenizer, split='train', shuffle=True)
        test = DACGdataLoader(args, tokenizer, split='test', shuffle=False)
        if args.smoke:
            train, test = SmokeLoader(train, 2, args.max_seq_length), SmokeLoader(test, 1)
        model = SEMDRG(args, tokenizer)
        optimizer = build_optimizer(args, model)
        scheduler = build_lr_scheduler(args, optimizer)
        trainer = FoldTrainer(model, compute_loss, compute_scores, optimizer, args, scheduler, train, test, provenance)
        trainer.train()
