import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


class AERPP(nn.Module):
    def __init__(self, cfg, args):
        super(AERPP, self).__init__()
        self.cfg = cfg
        self.args = args
        if self.cfg.embedding_dims != 0:
            self.embedding = nn.Embedding(self.cfg.event_classes, self.cfg.embedding_dims)
            self.dropout = nn.Dropout(self.cfg.dropout)
            self.lstm = nn.LSTM(self.cfg.embedding_dims + 1, self.cfg.lstm_hidden_unit_dims,
                                self.cfg.lstm_layers)
        else:
            self.lstm = nn.LSTM(self.cfg.event_classes + 1, self.cfg.lstm_hidden_unit_dims,
                                self.cfg.lstm_layers)
        self.attn = nn.Linear(self.cfg.lstm_hidden_unit_dims, self.cfg.event_classes)
        self.mlp = nn.Linear(self.cfg.lstm_hidden_unit_dims, self.cfg.mlp_dims)
        self.event_linear = nn.Linear(self.cfg.mlp_dims, 1)
        self.time_linear = nn.Linear(self.cfg.mlp_dims, 1)

    def forward(self, input, length):
        time_sequences = torch.tensor(input[:, :, 0:1], dtype=torch.float, device=self.args.device)
        event_sequences = torch.tensor(input[:, :, 1:], dtype=torch.long, device=self.args.device)
        if self.cfg.embedding_dims != 0:
            event_embedding = self.embedding(event_sequences)
            event_embedding_dropout = self.dropout(event_embedding)
            time_event_input = torch.cat((time_sequences, event_embedding_dropout), 2)
        else:
            event_one_hot = torch.zeros(event_sequences.shape[0],
                                        event_sequences.shape[1],
                                        self.cfg.event_classes,
                                        dtype=torch.float,
                                        device=self.args.device).scatter_(2, event_sequences, 1.0)
            time_event_input = torch.cat((time_sequences, event_one_hot), 2)
        time_event_input_packed = nn.utils.rnn.pack_padded_sequence(time_event_input,
                                                                    length,
                                                                    batch_first=True)
        h0 = torch.zeros(self.cfg.lstm_layers,
                         time_event_input.shape[0],
                         self.cfg.lstm_hidden_unit_dims,
                         dtype=torch.float,
                         device=self.args.device,
                         requires_grad=True)
        c0 = torch.zeros(self.cfg.lstm_layers,
                         time_event_input.shape[0],
                         self.cfg.lstm_hidden_unit_dims,
                         dtype=torch.float,
                         device=self.args.device,
                         requires_grad=True)
        output_packed, hidden = self.lstm(time_event_input_packed, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        attn_weight = torch.abs(torch.tanh(self.attn(output)))
        # attn_weight.data[attn_weight.data < self.cfg.THRESHOLD] = 0.0
        attn_weight = attn_weight.permute(0, 2, 1).contiguous()
        attn_applied = torch.bmm(attn_weight, output)
        output_mlp = torch.sigmoid(self.mlp(attn_applied))
        event_output = self.event_linear(output_mlp).squeeze()
        event_output_log_softmax = F.log_softmax(event_output, dim=1)
        selected_event = torch.argmax(event_output_log_softmax, dim=1)
        selected_attn_applied = torch.cat([
            output_mlp[i, selected_event[i], :].unsqueeze(0) for i in range(selected_event.shape[0])
        ], 0)
        time_output = self.time_linear(selected_attn_applied)
        return time_output, event_output_log_softmax


class AERPPLoss(nn.Module):
    def __init__(self, cfg, args):
        super(AERPPLoss, self).__init__()
        self.cfg = cfg
        self.args = args
        self.time_loss = nn.MSELoss()
        event2index = pickle.load(open(self.cfg.event_index_file, 'rb'))
        event_stat = np.zeros(len(event2index), dtype=np.float32)
        for event in event2index.values():
            event_stat[event['index']] = event['cnt']
        event_stat = event_stat.sum() / event_stat
        event_stat = torch.from_numpy(event_stat)
        event_stat = event_stat.to(args.device)
        self.event_loss = nn.NLLLoss(weight=event_stat)

    def forward(self, output, target):
        time_target = torch.tensor(target[:, 0], dtype=torch.float, device=self.args.device)
        event_target = torch.tensor(target[:, 1], dtype=torch.long, device=self.args.device)
        time = self.time_loss(output[0], time_target)
        event = self.event_loss(output[1], event_target)
        return time, event, self.cfg.loss_alpha * time + event
