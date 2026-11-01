import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='SEM-DRG')
    parser.add_argument('--fold', '-fold', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--split_root', type=Path, default=ROOT / 'data/splits')
    parser.add_argument('--ann_path', type=Path)
    parser.add_argument('--image_dir', type=Path, default=ROOT / 'data/images')
    parser.add_argument('--entity_dir', type=Path, default=ROOT / 'assets/entity_features')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--run', default='run_01')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', choices=['auto'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--print_config', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=160, help='max sequence length of reports')
    parser.add_argument('--threshold', type=int, default=3, help='cut off frequency for words')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=12)
    parser.add_argument('--report_interval', type=int, default=5, help='every X epochs, export test reports as Sample_epoch{epoch}.json')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.max_seq_length < 2:
        parser.error('epochs and batch_size must be positive; max_seq_length must be at least two')
    if args.d_model != 512 or args.d_vf != 2048:
        parser.error('SEM-DRG requires d_model=512 and d_vf=2048')
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('run must be a directory name')
    args.seed = 9232 + args.fold if args.seed is None else args.seed
    args.ann_path = str((args.ann_path or args.split_root / f'fold_{args.fold}/DAGC.json').resolve())
    args.image_dir = str(args.image_dir.resolve())
    args.entity_dir = str(args.entity_dir.resolve())
    args.split_root = str(args.split_root.resolve())
    args.output_dir = str(args.output_dir.resolve())
    args.save_dir = str(Path(args.output_dir) / f'fold_{args.fold}' / f'seed_{args.seed}' / (('smoke_' if args.smoke else '') + args.run))
    args.record_dir = args.save_dir
    args.dataset_name = 'df'
    args.train_split = 'train'
    args.visual_extractor = 'resnet101'
    args.visual_extractor_pretrained = True
    args.n_gpu = 1
    args.save_period = 1
    args.monitor_mode = 'max'
    args.optim = 'AdamW'
    args.lr_ve = 2e-05
    args.lr_ed = 0.0002
    args.weight_decay = 0.01
    args.amsgrad = True
    args.lr_scheduler = 'CosineAnnealingLR'
    args.cosine_t_max = args.epochs
    args.cosine_eta_min = 1e-07
    return args

def main():
    args = parse_args()
    if args.print_config:
        print(json.dumps(vars(args), indent=2))
        return
    from modules.fold_runner import run_fold
    run_fold(args)
if __name__ == '__main__':
    main()
