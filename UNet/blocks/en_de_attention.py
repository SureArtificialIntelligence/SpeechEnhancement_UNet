import torch
import torch.nn as nn
import math


class AttentionEnDe(nn.Module):
    def __init__(self, pars, att_direction):
        super(AttentionEnDe, self).__init__()
        self.convQ = nn.Conv2d(**pars['Q'])
        self.convK = nn.Conv2d(**pars['K'])
        self.convV = nn.Conv2d(**pars['V'])

        self.att_direction = att_direction
        self.num_heads = pars['num_heads']

        self.softmax = nn.Softmax(dim=-1)
        self.convO = nn.Conv2d(**pars['O'])

    def forward(self, key, value, query):
        # key: bs x ch1 x F x T, query: bs x ch2 x F x T
        q = self.convQ(query)  # [bs, ch, F, T]
        k = self.convK(key)
        v = self.convV(value)

        if self.att_direction == 't':
            pass
        elif self.att_direction == 'f':
            q = q.permute(0, 1, 3, 2)  # [bs, ch, T, F]
            k = k.permute(0, 1, 3, 2)
            v = v.permute(0, 1, 3, 2)
        else:
            print('Attention direction error!')

        q_bs, q_ch, q_keep, q_attn = q.size()
        k_bs, k_ch, k_keep, k_attn = k.size()
        v_bs, v_ch, v_keep, v_attn = v.size()

        q = torch.reshape(q, [q_bs, q_ch * q_keep, q_attn]).permute(0, 2, 1)   # [bs, attn, ch*keep]     d = ch * keep
        k = torch.reshape(k, [k_bs, k_ch * k_keep, k_attn]).permute(0, 2, 1)
        v = torch.reshape(v, [v_bs, v_ch * v_keep, v_attn]).permute(0, 2, 1)

        # split into multi-heads
        bs = q_bs
        d_sub = q.size()[-1] // self.num_heads
        Q, K, V = [element.view(bs, -1, self.num_heads, d_sub).transpose(1, 2)
                   for element in [q, k, v]]  # [bs, num_heads, attn, d_sub]
        attn_V = self.attention(Q, K, V).transpose(1, 2).contiguous().view(bs, -1, d_sub * self.num_heads).transpose(1, 2)  # [bs, d, attn]
        attn_V = torch.reshape(attn_V, [v_bs, v_ch, v_keep, v_attn])

        # weight = torch.matmul(q, k)  # T x T
        # attn = self.sm(weight)
        # attn_value = torch.matmul(attn, v)
        # attn_value = torch.reshape(attn_value, [v_bs, v_ch, v_keep, v_attn])

        if self.att_direction == 'f':
            attn_V = attn_V.permute(0, 1, 3, 2)
        out = self.convO(attn_V)
        return out

    def attention(self, query, key, value, mask=None, padding_mask=None):
        """
        scores = Q * K^T => [bs, tgt_T, T]    scores * V => [bs, tgt_T, E]
        :param Q: [bs, tgt_T, E]
        :param K: [bs, T, E]
        :param V: [bs, T, E]
        :param mask:  [bs, tgt_T, T] (equals to scores)
        :param padding_mask:  [bs, tgt_T, T] (equals to scores)
        :return: attentive values scored by query*key^T
        """

        num_heads, d_sub = query.size(1), query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_sub)   # [bs, num_heads, attn, attn]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)  # [bs, h, tgt_T, T]
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))
        scores = self.softmax(scores)
        # get rid of inplace operation (scores[scores != scores] = 0.)
        temp = scores.clone()
        temp[scores != scores] = 0.
        scores = temp
        attn_V = torch.matmul(scores, value)    # scores: [bs, num_heads, attn, attn], value: [bs, num_heads, attn, d_sub] => [bs, num_heads, attn, d_sub]
        # self.write_doc(['scores', 'mh_V'], [scores, attn_V], self.v)
        return attn_V



class SEBlock(nn.Module):
    def __init__(self, pars):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(**pars['avg_pool'])
        self.fc1 = nn.Linear(**pars['fc1'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(**pars['fc2'])
        self.act = nn.Sigmoid()

    def forward(self, x):
        ch_attn = self.global_pool(x)[:, :, 0, 0]
        ch_attn = self.act(self.fc2(self.relu(self.fc1(ch_attn))))
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)
        attn_x = ch_attn * x
        return attn_x


if __name__ == '__main__':
    dummy_en = torch.rand([10, 90, 24, 4])  # [bs, ch, F, T]
    dummy_de = torch.rand([10, 90, 24, 4])  # [bs, ch, F, T]

    pars_att = {
        'num_heads': 3,
        'Q': {
            'in_channels': 90,
            'out_channels': 180,
            'kernel_size': (1, 1)
        },
        'K': {
            'in_channels': 90,
            'out_channels': 180,
            'kernel_size': (1, 1)
        },
        'V': {
            'in_channels': 90,
            'out_channels': 180,
            'kernel_size': (1, 1)
        },
        'O': {
            'in_channels': 180,
            'out_channels': 90,
            'kernel_size': (1, 1)
        },
    }

    r = 10
    pars_se = {
        'avg_pool': {
            'output_size': (1, 1)
        },
        'fc1': {
            'in_features': 90,
            'out_features': int(90/r)
        },
        'fc2': {
            'in_features': int(90/r),
            'out_features': 90
        }
    }

    att_f = AttentionEnDe(pars_att, 'f')
    att_t = AttentionEnDe(pars_att, 't')
    att_output = att_t(dummy_en, dummy_en, dummy_de)
    att_output = att_f(att_output, att_output, dummy_de)
    print(att_output.shape)

    att_se = SEBlock(pars_se)
    att_se_output = att_se(att_output)
    print(att_se_output.shape)
