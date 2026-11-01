import json
import re
import unicodedata
from collections import Counter

class Tokenizer(object):

    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        self.clean_report = self.clean_report_dagc
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)
        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = ({}, {})
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return (token2idx, idx2token)

    def clean_report_dagc(self, report: str) -> str:
        if report is None:
            return ''
        t = str(report)
        t = unicodedata.normalize('NFKC', t)
        t = re.sub('[\\u200B-\\u200F\\u202A-\\u202E\\u2060\\uFEFF]', '', t)
        t = t.replace('\xa0', ' ').replace('\u202f', ' ')
        t = t.replace('—', '-').replace('–', '-')
        t = t.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        t = re.sub('\\*(.*?)\\*', '\\1', t)
        t = re.sub('(\\d+)\\s*[mM][lL]\\b', '\\1 mL', t)
        t = re.sub('(\\d+)\\s*[lL]\\b', '\\1 L', t)
        t = re.sub('\\b[mM][lL]\\b', 'mL', t)
        t = re.sub('\\b[lL]\\b', 'L', t)
        t = re.sub('[^A-Za-z0-9\\s\\.,!?;:\\(\\)\\[\\]\\-/%+\\’\\\'"]+', '', t)
        t = re.sub('\\s*/\\s*', '/', t)
        t = re.sub('\\s+', ' ', t).strip()
        t = re.sub('\\s*<unk>\\s*', ' ', t, flags=re.IGNORECASE)
        t = re.sub('\\s+', ' ', t).strip()
        if t and (not re.search('[.!?]$', t)):
            t += '.'
        return t

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
