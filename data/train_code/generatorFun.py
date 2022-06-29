
# Package Importing
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import h5py


LATENT_DIM = 128



def generator_1(num_fake_1, file_generator_1, file_real_1):
    use_cuda = torch.cuda.is_available()
    generator_C = AutoEncoder(48)
    checkpoint = torch.load(file_generator_1, map_location=torch.device('cpu'))
    generator_C.load_state_dict(checkpoint['model_states']['net'])
    generator_C = generator_C.cuda()
    generator_C.eval() 
    num_tx = 32
    num_rx = 4
    num_delay = 32
    latent_dim = 128
    size_packet = 500
    with torch.no_grad():
        for idx in range(int(num_fake_1 / size_packet)):
            latent_vectors = np.random.randint(2, size=(size_packet, 48))
            latent_vectors = torch.from_numpy(latent_vectors)
            latent_vectors = latent_vectors.cuda()

            output = generator_C.decoder(latent_vectors)  # bx4x32x32x2

            output = output.detach().cpu().numpy()
            if idx == 0:
                data_fake_all = output
            else:
                data_fake_all = np.concatenate([data_fake_all, output], axis=0)

    # data_fake_all_copy = data_fake_all.copy()
    # data_fake_all_copy = data_fake_all_copy.reshape(num_fake_1 * 4, -1)
    # data_var = np.var(data_fake_all_copy, axis=1)
    # epsilon = sorted(data_var,reverse=True)[num_fake_1]
    # choice = data_var >= epsilon
    #
    # data_fake_all = data_fake_all[choice][:num_fake_1]

    data_fake_all = data_fake_all[:, :, :, :, 0] + data_fake_all[:, :, :, :, 1] * 1j

    return data_fake_all


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




def generator_2(num_fake_2, file_generator_2, file_real_2):
    use_cuda = torch.cuda.is_available()
    generator_C = AutoEncoder(48)
    checkpoint = torch.load(file_generator_2, map_location=torch.device('cpu'))
    generator_C.load_state_dict(checkpoint['model_states']['net'])
    generator_C = generator_C.cuda()
    generator_C.eval()

    # 读取真实数据
    # h_true_smp = np.load(file_real_1, allow_pickle=True)
    NUM_REAL_2 = 4000
    NUM_SAMPLE_TRAIN = 4000
    NUM_RX = 4
    NUM_TX = 32
    NUM_DELAY = 32
    # real_1_test = h5py.File(file_real_2, 'r')
    # real_1_test = np.transpose(real_1_test['H2_32T4R'][:])
    # real_1_test = real_1_test[::int(real_1_test.shape[0] / NUM_REAL_2), :, :, :]
    # real_1_test = real_1_test[:, :, :, :, np.newaxis]
    # real_1_test = np.concatenate([real_1_test['real'], real_1_test['imag']], 4)
    
    # Augdata = []
    # for i in range(200):
    #     temp = real_1_test[20 * i:20 * (i + 1), :, :, :].copy()
    #     count = 0
    #     for j in range(20):
    #         for k in range(j, 20):
    #             for r in range(5, 6):
    #                 comb = r / 10 * temp[j, :, :, :] + (1 - r / 10) * temp[k, :, :, :]
    #                 comb.astype(np.float32)
    #                 Augdata.append(comb)
    #                 count += 1
    #             if count==200:
    #                 break
    #         if count == 200:
    #             break
    #
    # Augdata = np.array(Augdata)
    # Augdata = norm_data(Augdata, Augdata.shape[0], NUM_RX, NUM_TX, NUM_DELAY)
    # fake_data = np.reshape(Augdata, [Augdata.shape[0], NUM_RX, NUM_TX, NUM_DELAY, 2])
    # fake_data_r = fake_data[:, :, :, :, 0]
    # fake_data_i = fake_data[:, :, :, :, 1]
    # data_fake_all = fake_data_r + fake_data_i * 1j

    # real_1_test = np.reshape(real_1_test, [NUM_SAMPLE_TRAIN, NUM_RX * NUM_TX, NUM_DELAY * 2, 1])
    # true_data = norm_data(real_1_test, NUM_SAMPLE_TRAIN, NUM_RX, NUM_TX, NUM_DELAY)
    
    num_tx = 32
    num_rx = 4
    num_delay = 32
    latent_dim = 128
    size_packet = 500

    
    with torch.no_grad():
        for idx in range(int(num_fake_2*1.5 / size_packet)):
            latent_vectors = np.random.randint(2, size=(size_packet, 48))
            latent_vectors = torch.from_numpy(latent_vectors)
            latent_vectors = latent_vectors.cuda()

            output = generator_C.decoder(latent_vectors)  # bx4x32x32x2

            output = output.detach().cpu().numpy()
            if idx == 0:
                data_fake_all = output
            else:
                data_fake_all = np.concatenate([data_fake_all, output], axis=0)

    data_fake_all_copy = data_fake_all.copy()
    data_fake_all_copy = data_fake_all_copy.reshape(int(num_fake_2 * 1.5), -1)
    data_var = np.var(data_fake_all_copy, axis=1)
    epsilon = sorted(data_var,reverse=True)[num_fake_2]
    choice = data_var >= epsilon

    data_fake_all = data_fake_all[choice][:num_fake_2]
    # print(data_fake_all.shape)

    data_fake_all = data_fake_all[:, :, :, :, 0] + data_fake_all[:, :, :, :, 1] * 1j

    return data_fake_all

# data_fake_all = generator_2(4000, 'generator_2.pth.tar', ' ')
# data_fake_all = generator_1(4000, 'generator_1.pth.tar', ' ')
# generator_C = Decoder(feedback_bits=48, dropout=0.1)
# generator_C.load_state_dict(torch.load('generator_2.pth.tar')['state_dict'])
# data_fake_all = generator_1(500, 'generator_1.pth.tar', ' ')