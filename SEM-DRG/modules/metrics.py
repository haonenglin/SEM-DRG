from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

def compute_scores(gts, res):
    scorers = [(Bleu(4), ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']), (Meteor(), 'METEOR'), (Rouge(), 'ROUGE_L')]
    eval_res = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
