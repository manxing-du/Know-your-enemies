import torch
from torch import nn
import torch.nn.functional as F
from util_funcs import *
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, '../')

import master_config as config
import os.path as osp
import os
from TransformerModels import Encoder, Decoder, PositionalEncoding

torch.manual_seed(config.seed)
seed=config.seed

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, input):
        reshaped_input = input.view(input.size(0), -1)
        return reshaped_input

class NeuralNet(nn.Module):
    def __init__(self, vocal_size, num_features, num_outputs, h1, h2,
                 emb_dropout, lin_dropout):
        super(NeuralNet, self).__init__()
        embedding_dim = int(vocal_size ** 0.25)
        self.layers = nn.Sequential(
            nn.Embedding(vocal_size, embedding_dim),
            Flatten(),
            nn.Dropout(emb_dropout),
            nn.Linear(embedding_dim * num_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(True),
            nn.Dropout(lin_dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(True),
            nn.Dropout(lin_dropout),
            nn.Linear(h2, num_outputs),
            nn.Sigmoid()
            #nn.Softmax(dim=1)
        )
        self.layers.apply(init_weights)

    def forward(self, x):
        out = self.layers(x)
        return out

class Transformer(nn.Module):
    def __init__(self, vocal_size, num_features, num_outputs,
                        model_dim, drop_prob, point_wise_dim, num_sublayer, num_head, is_cuda):
        super().__init__()
        self.embedding_dim = int(vocal_size ** 0.25)
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.model_dim = model_dim

        # output torch.Size([128, 301])
        self.embedding = nn.Embedding(vocal_size, self.embedding_dim)

        # Middle layer size is model_dim - 1
        self.middle_layer = nn.Linear(self.embedding_dim * num_features, model_dim)

        self.encoder_sublayers = nn.Sequential(
            *[Encoder.EncoderSubLayer(model_dim, num_head, drop_prob, point_wise_dim) for _ in range(num_sublayer)]
        )
        self.encode_pos_encoder = PositionalEncoding.PositionalEncoding(model_dim, drop_prob, is_cuda)

        # self.decoder_sublayers = nn.ModuleList([
        #     Decoder.DecoderSubLayer(model_dim, num_head, drop_prob, point_wise_dim) for _ in range(num_sublayer)
        # ])

        self.linear = nn.Linear(self.model_dim, 1)
        #
        # self.decoder_mask =  np.logical_not(np.triu(np.ones((1, num_outputs, num_outputs)),
        #                                           k=1).astype('uint8')).astype('uint8')
        #
        # if is_cuda:
        #   self.decoder_mask = torch.from_numpy(self.decoder_mask).cuda()
        #   self.source_mask = torch.from_numpy(np.ones((1, num_outputs, num_outputs))).cuda()
        # else:
        #   self.decoder_mask = torch.from_numpy(self.decoder_mask)
        #   self.source_mask = torch.from_numpy(np.ones((1, num_outputs, num_outputs)))

    def forward(self, x):
        batch_size = x.size(0)
        emb_x = self.embedding(x)
        input = emb_x.view(batch_size, self.num_features * self.embedding_dim)
        middle_layer = F.relu(self.middle_layer(input))

        # Not using DRSA encoding and it repeat inside element so there no need for mapping.
        # (~60% confidence)
        input_x = middle_layer.unsqueeze(1).repeat(1, self.num_outputs, 1)
        # encoded_input = self.encoder_sublayers(input_x)

        encoded_inp_pos = self.encode_pos_encoder(input_x)
        encoded_input = self.encoder_sublayers(encoded_inp_pos)

        outputs = encoded_input
        # for sub in self.decoder_sublayers:
        #     outputs = sub(encoded_input, outputs, self.source_mask, self.decoder_mask)

        new_output = outputs.view(self.num_outputs * batch_size, self.model_dim)
        preds = torch.transpose(torch.sigmoid(self.linear(new_output)), 0, 1)[0]
        return preds.view(batch_size, self.num_outputs)


class opponent(object):
    def __init__(self, alpha, beta, epochs, num_outputs, h1, h2, vocal_size,
                 model_path, camp, sample_size, num_features, emb_dropout, lin_dropout, c0, reward, agent_num,
                 agent_index, use_cuda, op):

        self.alpha = alpha
        self.beta = beta

        self.num_outputs = num_outputs
        self.h1 = h1
        self.h2 = h2
        self.epochs = epochs
        self.model_path = model_path
        self.camp = camp
        self.sample_size = sample_size

        # Initialize data generator
        # self.training_generator = training_generator
        self.vocal_size = vocal_size
        self.num_features = num_features
        self.emb_dropout = emb_dropout
        self.lin_dropout = lin_dropout


        # Initialize model
        self.op = op
        if self.op == 'ffn':
            self.model = NeuralNet(self.vocal_size, self.num_features, self.num_outputs, self.h1, self.h2, self.emb_dropout,
                                   self.lin_dropout)

        else:
            # Just replace here.
            # self.model = Transformer(self.vocal_size, self.num_features, self.num_outputs, 128, 0.1, 512, 1, 8, use_cuda)
            self.model = Transformer(self.vocal_size, self.num_features, self.num_outputs, 16, 0.1, 64, 1, 2, use_cuda)

        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

        self.best_model_index = 0

        self.c0 = c0
        self.reward = reward
        self.agent_num = agent_num
        self.agent_index = agent_index

        self.use_cuda = use_cuda

    def train(self, training_generator, val_generator, file_prefix, camp):
        train_loss = []
        val_loss = []
        val_anlp = []
        counter = 0
        try:
            for epoch in range(self.epochs):
                print('start epoch %d' % epoch)
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train(True)
                        running_loss = 0.0
                        n_batch = 0

                        for x_batch, c_batch,  b_batch,  m1_batch, m2_batch in training_generator:
                            n_batch += 1
                            # x_batch = get_onehot_data(x_batch, self.vocal_size)
                            if self.use_cuda:
                                x_batch, c_batch, b_batch, m1_batch, m2_batch = Variable(x_batch.cuda()), \
                                                                                Variable(c_batch.cuda()), \
                                                                                Variable(b_batch.cuda()), \
                                                                                Variable(m1_batch.cuda()), \
                                                                                Variable(m2_batch.cuda())
                                x_batch = x_batch.type(torch.cuda.LongTensor)

                            else:
                                x_batch, c_batch, b_batch,  m1_batch, m2_batch = Variable(x_batch), Variable(c_batch), \
                                                                      Variable(b_batch), Variable(m1_batch), \
                                                                      Variable(m2_batch)
                                x_batch = x_batch.type(torch.LongTensor)

                            self.optimizer.zero_grad()
                            ypred_batch = self.model(x_batch)
                            loss = loss_full(c_batch, b_batch, m1_batch, m2_batch, ypred_batch, self.beta)
                            loss.backward()
                            self.optimizer.step()
                            running_loss += loss.item()
                            # print(loss)

                            # if n_batch % 200 == 0:  # print every 200 mini-batches just to print it out
                            #     print('[%d, %5d] loss: %.3f' %
                            #           (epoch, i + 1, running_loss / 200))
                        train_loss.append(np.log(running_loss / n_batch))
                            # running_loss = 0.0

                        save_path = self.model_path

                        if not osp.exists(save_path):
                            os.makedirs(save_path)

                        torch.save(self.model.state_dict(), save_path + self.camp + '_train_sample' + str(self.sample_size)
                                   + 'epoch_' + str(epoch) + 'lr_' + str(self.alpha) + file_prefix + '.pt')

                    else:
                        # self.model.train(False)
                        self.model.eval()
                        running_loss = 0.0
                        n_batch = 0
                        anlp = 0

                        for x_batch, c_batch,  b_batch,  m1_batch, m2_batch in val_generator:
                            batch_size = x_batch.shape[0]
                            n_batch += 1

                            if self.use_cuda:
                                x_batch, c_batch, b_batch, m1_batch, m2_batch = Variable(x_batch.cuda()), \
                                                                                Variable(c_batch.cuda()), \
                                                                                Variable(b_batch.cuda()), \
                                                                                Variable(m1_batch.cuda()), \
                                                                                Variable(m2_batch.cuda())
                                x_batch = x_batch.type(torch.cuda.LongTensor)

                            else:
                                x_batch, c_batch, b_batch, m1_batch, m2_batch = Variable(x_batch), \
                                                                                Variable(c_batch), \
                                                                                Variable(b_batch), \
                                                                                Variable(m1_batch), \
                                                                                Variable(m2_batch)
                                x_batch = x_batch.type(torch.LongTensor)

                            ypred_valbatch = self.model(x_batch)
                            loss = loss_full(c_batch, b_batch, m1_batch, m2_batch, ypred_valbatch, self.beta)
                            running_loss += loss.item()

                            hz = torch.sum(ypred_valbatch * m1_batch, dim=1)
                            hzp = torch.prod(1 - ypred_valbatch * m2_batch, dim=1)
                            score = torch.sum(torch.log(hz * hzp))
                            anlp += - score.item() / batch_size

                        val_anlp.append(anlp / n_batch)
                        val_loss.append(np.log(running_loss / n_batch))
                        print(val_loss)
                        min_delta = 0.01
                        print('validation anlp:')
                        print(val_anlp)
                        # if len(val_anlp) > 1:
                            # # compare the anlp results
                            # if val_anlp[-2] - val_anlp[-1] <= min_delta:
                            #     self.best_model_index = epoch - 1
                            #     print('early stopping at %d' % epoch)
                            #     raise StopIteration
                            # elif val_loss[-2] - val_loss[-1] <= min_delta:
                            #     self.best_model_index = epoch - 1
                            #     counter += 1
                            #     if counter == 3:
                            #         print('early stopping at %d' % epoch)
                            #         raise StopIteration
                            # else:
                            #     self.best_model_index = 5
                        self.best_model_index = np.argmin(val_anlp)

        except StopIteration:
            pass

        print('the best model is from epoch %d' % self.best_model_index)
        index_file_path = '../log/best_opp_model_index/agent_' + str(self.agent_index) + '/'

        if not osp.exists(index_file_path):
            os.makedirs(index_file_path)

        tmp_file = open(index_file_path + camp + '_best_model' + file_prefix + '.txt', 'w')
        tmp_file.write('%d\n' % self.best_model_index)
        tmp_file.close()
        return train_loss, val_loss, val_anlp

    def test(self, test_generator, best_model_index, c0, file_prefix, mode='test'):
        self.model.load_state_dict(torch.load(self.model_path + self.camp + '_train_sample' + str(self.sample_size)
                                              + 'epoch_' + str(best_model_index) + 'lr_' + str(self.alpha)
                                              + file_prefix + '.pt'))
        self.model.eval()
        anlp = 0
        nbatch = 0

        prediction_path = '../predictions/agent_' + str(self.agent_index) + '/'

        if not osp.exists(prediction_path):
            os.makedirs(prediction_path)

        f = open(prediction_path + self.camp + '_' + mode + file_prefix + '.txt', 'w')
        print(prediction_path + self.camp + '_' + mode + file_prefix + '.txt')

        for x_batch in test_generator:
            nbatch += 1
            # x_batch = get_onehot_data(x_batch, vocal_size)
            if self.use_cuda:
                x_batch = Variable(x_batch.cuda())
                x_batch = x_batch.type(torch.cuda.LongTensor)
            else:
                x_batch = Variable(x_batch)
                x_batch = x_batch.type(torch.LongTensor)

            ypred_testbatch = self.model(x_batch)

            # write the prediction to a file
            batch_list = ypred_testbatch.tolist()

            # calculate the p(z | x, \theta)
            for item in batch_list:
                n_event = [1-x for x in item]
                for i in range(len(item)):
                    if i == 0:
                        f.write('%.6f ' % item[i])
                    else:
                        pz = item[i] * np.prod(n_event[:i])
                        f.write('%.6f ' % pz)
                f.write('\n')

            # hz = torch.sum(ypred_testbatch * m1_batch, dim=1)
            # hzp = torch.prod(1 - ypred_testbatch * m2_batch, dim=1)
            # score = torch.sum(torch.log(hz * hzp))
            # anlp += - score.item() / x_batch.shape[0]
        # test_anlp = anlp / nbatch
        f.close()
        # return test_anlp
