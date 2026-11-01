import torch.nn as nn
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder_DER_MOE

class SEMDRG(nn.Module):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder_DER_MOE = EncoderDecoder_DER_MOE(args, tokenizer)

    def forward(self, images, targets=None, mode='train', ent_lists=None):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            return self.encoder_decoder_DER_MOE(images, fc_feats, att_feats, targets, mode='forward', ent_lists=ent_lists)
        if mode == 'sample':
            output, _ = self.encoder_decoder_DER_MOE(images, fc_feats, att_feats, mode='sample', ent_lists=ent_lists)
            return output
        raise ValueError(f'Unknown mode: {mode}')
