import torch
import torch.nn as nn
from TransformerModels import Attention, BetweenSubLayer, PointwiseFeedForward
from TransformerModels import PreprocessInput

class DecoderSubLayer(nn.Module):
    def __init__(self, model_dim, num_head, drop_prob, pointWise_dim):
        super().__init__()
        self.model_dim = model_dim

        self.masked_multi_headed = Attention.MultiHeadAttention(model_dim, num_head)
        self.in_between = BetweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

        self.multi_headed_encode = Attention.MultiHeadAttention(model_dim, num_head)
        self.in_between2 = BetweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

        self.point_wise = PointwiseFeedForward.PointWiseFeedForward(model_dim, pointWise_dim)
        self.in_between3 = BetweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

    def forward(self, encoded, inp, encode_mask, target_mask):
        first_layer = self.in_between(
            self.masked_multi_headed(inp, inp, inp, mask=target_mask),
            inp
        )

        second_layer = self.in_between2(
            self.multi_headed_encode(first_layer, encoded, encoded, mask=encode_mask),
            first_layer
        )

        last_layer = self.in_between3(
            self.point_wise(second_layer),
            second_layer
        )

        return last_layer

class Decoder(nn.Module):
    def __init__(self, vocab_size, output_size, model_dim, num_head, drop_prob, pointWise_dim, num_sublayer, is_cuda):
        super().__init__()
        self.preprocess = PreprocessInput.PreprocessInput(vocab_size, model_dim, drop_prob, is_cuda)
        self.num_sublayer = num_sublayer
        self.model_dim = model_dim

        self.all_sublayers = [
            DecoderSubLayer(model_dim, num_head, drop_prob, pointWise_dim)
            for _ in range(num_sublayer)
        ]

        self.final_linear = nn.Linear(model_dim, output_size)

    def forward(self, encoded, target, encode_mask, target_mask):
        inp_target = self.preprocess(target)
        for sub in self.all_sublayers:
            inp_target = sub(encoded, inp_target, encode_mask, target_mask)

        return self.final_linear(inp_target)
