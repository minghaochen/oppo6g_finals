import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from evaluation_training import K_nearest

from dataset import get_data
from torch.autograd import Variable


def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=256):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class ResidualNetwork(nn.Module):
    def __init__(self):
        super(ResidualNetwork, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), stride=(1,), padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=(3, 3), stride=(1,), padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=(3, 3), stride=(1,), padding=1),
            nn.ReLU(),
        )

    def forward(self, input):
        output = input.unsqueeze(1)
        output = self.conv_block(output)
        output = output.squeeze()
        output = output + input
        return output


class AttentionLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(AttentionLayer, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.temperature = d_k ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_input, mask=None):

        q,k,v = enc_input,enc_input,enc_input
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        q = torch.matmul(attn, v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        enc_output = self.layer_norm(q)

        residual = enc_output
        enc_output = self.w_2(F.relu(self.w_1(enc_output)))
        enc_output = self.dropout(enc_output)
        enc_output += residual
        enc_output = self.layer_norm(enc_output)

        return enc_output

class Encoder(nn.Module):
    def __init__(self, n_position=256, dropout=0.1, feedback_bits=512, B=2):
        super(Encoder, self).__init__()

        self.feedback_bits = feedback_bits
        self.B = B

        self.position_dec = PositionalEncoding(32, n_position=n_position)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(32, eps=1e-6)

        self.trans_encoder = nn.Sequential(
            ResidualNetwork(),
            AttentionLayer(32, 2048, 8, 32, 32, dropout=dropout),
            AttentionLayer(32, 2048, 8, 32, 32, dropout=dropout),
        )
        self.fc = nn.Linear(1 * 32 * 256, int(self.feedback_bits // self.B))
        self.out_bn = nn.BatchNorm1d(int(self.feedback_bits // self.B))

        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(B)

    def forward(self, input):
        output = input.permute((0, 3, 1, 2, 4)).reshape(-1, 32, 256).permute((0, 2, 1))

        enc_output = self.dropout(self.position_dec(output))
        enc_output = self.layer_norm(enc_output)
        output = self.trans_encoder(enc_output)

        out = self.out_bn(self.fc(output.view(-1, 32 * 256)))
        out = self.sig(out)

        out = self.quantize(out)

        return out


class Decoder(nn.Module):
    def __init__(self, n_position=256, drop_out=0.1, feedback_bits=512, B=2):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.B = B

        self.position_dec = PositionalEncoding(32, n_position=n_position)
        self.dropout = nn.Dropout(p=drop_out)
        self.layer_norm = nn.LayerNorm(32, eps=1e-6)
        self.trans_decoder = nn.Sequential(
            ResidualNetwork(),
            AttentionLayer(32, 2048, 8, 32, 32, dropout=drop_out),
            AttentionLayer(32, 2048, 8, 32, 32, dropout=drop_out),
        )
        self.fc = nn.Linear(int(self.feedback_bits // self.B), 1 * 32 * 256)

        self.sig = nn.Sigmoid()
        self.dequantize = DequantizationLayer(self.B)

    def forward(self, input):
        bits = self.dequantize(input)
        bits = torch.log(bits / (1.0 - bits))

        bits = torch.cat([bits], dim=1)
        csi_image = self.fc(bits).reshape(-1, 256, 32)

        enc_output = self.dropout(self.position_dec(csi_image))
        enc_output = self.layer_norm(enc_output)
        dec_output = self.trans_decoder(enc_output)

        out = dec_output.permute((0, 2, 1)).reshape((-1, 32, 4, 32, 2)).permute((0, 2, 3, 1, 4))
        out = self.sig(out) - 0.5

        return out


class AutoEncoder(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, feedback_bits=256):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(feedback_bits=feedback_bits) # dropout需要调整
        self.decoder = Decoder(feedback_bits=feedback_bits)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        output = self.decoder(z)
        return output


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def multistep_lr_decay(optimizer, current_step, schedules):
    """Manual LR scheduler for implementing schedules described in the WAE paper."""
    for step in schedules:
        if current_step == step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / schedules[step]

    return optimizer


class SmiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss().cuda()

    def forward(self, output, input, epoch=0):
        input = input.reshape((-1, 4 * 32 * 32, 2))
        input_complex = torch.complex(input[:, :, 0], input[:, :, 1])
        input = F.normalize(input_complex, p=2, dim=1)
        output = output.reshape((-1, 4 * 32 * 32, 2))
        output_complex = torch.complex(output[:, :, 0], output[:, :, 1])
        output = F.normalize(output_complex, p=2, dim=1)
        sim = torch.abs(torch.sum(input * torch.conj(output), dim=1))
        loss = []
        sim_loss = 1.0 - torch.mean(sim * sim)

        loss.append(sim_loss)
        loss.append(self.mse(input.real, output.real))
        loss.append(self.mse(input.imag, output.imag))

        return loss


class Trainer(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.feedback_bits = args.feedback_bits
        self.num_sample = args.num_sample
        self.dataset = args.dataset

        self.net = cuda(AutoEncoder(self.feedback_bits), self.use_cuda)
        self.lr_schedules = {5000: 1, 10000: 1, 15000: 1, 20000: 1}
        self.change_lr = args.change_lr
        self.criterion = SmiLoss()
        self.optim = optim.AdamW(self.net.parameters(), lr=self.lr)

        self.ckpt_dir = args.ckpt_dir

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_model(self.ckpt_name)

        self.save_step = args.save_step
        self.display_step = args.display_step
        self.evaluate_step = args.evaluate_step
        self.batch_size = args.batch_size
        self.data_loader, self.all_data, self.all_normed_data = get_data(args.data_path, args.row_name, args.num_sample,
                                                                         args.batch_size)

    def train(self):
        self.net.train()
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        best_score = 1000
        while not out:
            for x in self.data_loader:
                x = x.float()
                self.global_iter += 1
                pbar.update(1)

                # self.optim = multistep_lr_decay(self.optim,
                #                                     self.global_iter / len(self.data_loader),
                #                                     self.lr_schedules)

                if self.global_iter == self.change_lr: # 200000
                    print('reduce lr!')
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = param_group['lr'] / 10

                x = Variable(cuda(x, self.use_cuda))
                x_recon = self.net(x)
                recon_loss_list = self.criterion(x, x_recon)
                recon_loss = sum(recon_loss_list[1:])
                self.optim.zero_grad()
                recon_loss.backward()
                self.optim.step()

                if self.global_iter % self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.8f} '.format(
                        self.global_iter, recon_loss.data.item()))

                if self.global_iter % self.save_step == 0:
                    self.save_model(f'vae_dataset_{self.dataset}_{self.global_iter}.pth.tar')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter % self.evaluate_step == 0:
                    sim, multi, score = self.evaluate_fake()
                    pbar.write('[{}] sim:{:.6f} multi:{:.6f} score:{:.6f}'.format(
                        self.global_iter, sim, multi, score))
                    if score < best_score:
                        best_score = score
                        self.save_model(f'generator_{self.dataset}.pth.tar')
                    print('best_score', best_score)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

    def evaluate_fake(self):
        self.net.eval()
        size_packet = 100
        fake_samples = None
        with torch.no_grad():
            for idx in range(self.num_sample // size_packet): # 500 4000 切换
                random_z = np.random.randint(2, size=(size_packet, self.feedback_bits))
                random_z = cuda(torch.from_numpy(random_z), self.use_cuda)
                fake = self.net.decoder(random_z).detach().cpu().numpy()
                fake = np.reshape(fake, [size_packet, 4, 32, 32, 2])
                fake_r = fake[:, :, :, :, 0]
                fake_i = fake[:, :, :, :, 1]
                fake_reshape = fake_r + fake_i * 1j
                if idx == 0:
                    fake_samples = fake_reshape
                else:
                    fake_samples = np.concatenate((fake_samples, fake_reshape), axis=0)
        sim, multi, score = K_nearest(self.all_data, fake_samples, 4, 32, 32, 1)
        self.net.train()

        return sim, multi, score

    def save_model(self, filename):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optim.state_dict(), }
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_model(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=2022, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=50000, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    parser.add_argument('--feedback_bits', default=48, type=int, help='dimension of the bits space')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    # change dataset
    parser.add_argument('--dataset', default=1, type=int, help='dataset')
    parser.add_argument('--data_path', default='data/H1_32T4R.mat', type=str, help='data source')
    parser.add_argument('--row_name', default='H1_32T4R', type=str, help='The row name')
    parser.add_argument('--num_sample', default=500, type=int, help='The number of samples')

    parser.add_argument('--display_step', default=100, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=2000, type=int,
                        help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--evaluate_step', default=500, type=int,
                        help='number of iterations after which for performance evaluation')
    parser.add_argument('--reconstruction_step', default=1000, type=int,
                        help='number of iterations after which for reconstruction virtualization')
    parser.add_argument('--do_virtualization', default=False, type=str2bool,
                        help='do reconstruction virtualization or not')

    parser.add_argument('--ckpt_dir', default='/workfolder/oppo6g_finals/data/user_data', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='load previous checkpoint. insert checkpoint filename')

    args = parser.parse_args()

    if args.dataset == 1:
        print("training dataset1")
        args.data_path = '../raw_data/H1_32T4R.mat'
        args.row_name = 'H1_32T4R'
        args.num_sample = 500
        args.max_iter = 160000
        args.change_lr = 160000 // 4 * 3
    else:
        print("training dataset2")
        args.data_path = '/workfolder/oppo6g_finals/data/raw_data/H2_32T4R.mat'
        args.row_name = 'H2_32T4R'
        args.num_sample = 4000
        args.max_iter = 250000
        args.change_lr = 250000 // 4 * 3

    trainer = Trainer(args)
    trainer.train()
    # trainer.show_reconstruction(5)
