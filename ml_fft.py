import torch
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt

# weight = torch.tensor([
#     [1., 0.],
#     [0.707, -0.707],
#     [0., -1.],
#     [-0.707, -0.707]
# ], requires_grad=True)


class FFTModel(torch.nn.Module):
    def __init__(self, len):
        super(FFTModel, self).__init__()

        if (np.log(len) / np.log(2)) % 1 != 0:
            raise Exception('Signal length must be power of 2')
        self.len = len
        # param_data = torch.rand([self.len // 2, 2])
        base_param = np.linspace(0, -3.1415, self.len // 2 + 1)
        cos_param = np.cos(base_param)[:self.len // 2]
        sin_param = np.sin(base_param)[:self.len // 2]
        param_data = np.concatenate([np.expand_dims(cos_param, -1), np.expand_dims(sin_param, -1)], axis=1)

        self.weight = torch.nn.Parameter(data=torch.tensor(param_data), requires_grad=True)

    def complex_mul(self, in_1, in_2):
        in_1_re = in_1[:, :, 0]
        in_1_im = in_1[:, :, 1]
        in_2_re = in_2[:, 0]
        in_2_im = in_2[:, 1]

        out_re = in_1_re * in_2_re - in_1_im * in_2_im
        out_im = in_1_re * in_2_im + in_1_im * in_2_re
        out = torch.cat([out_re.unsqueeze(-1), out_im.unsqueeze(-1)], -1)

        return out

    def forward(self, input):
        if input.shape[0] != self.len:
            raise Exception('Signal length does not match expected value')
        input = input.view(-1, 1)
        input = torch.cat([input, torch.zeros_like(input)], 1)
        x = input.view([-1, 1, 2])
        delimeter = x.shape[0] // 2
        while delimeter != 0:
            weight_ = self.weight[0::delimeter]
            output_1 = x[:delimeter] + self.complex_mul(x[delimeter:], weight_)
            output_2 = x[:delimeter] - self.complex_mul(x[delimeter:], weight_)
            x = torch.cat([output_1, output_2], 1)
            delimeter = delimeter // 2

        return x


def train_fft_model(x, expected_output, weights=None):
    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    x = torch.tensor(x).to(device)
    expected_output = torch.tensor(expected_output).to(device)
    fft_model = FFTModel(len(x)).to(device)
    if weights is not None:
        fft_model.weight.data = weights
    opt = torch.optim.Adam(fft_model.parameters(), 0.1)
    for i in range(2001):
        opt.zero_grad()
        output = fft_model(x)
        output = torch.sqrt(torch.sum(torch.pow(output, 2), axis=2))
        err = torch.mean(torch.pow(expected_output - output, 2))
        if i % 100 == 0:
            print(i, err)
        err.backward()
        opt.step()

    return fft_model


# print(model(x))
# exit()
if __name__ == '__main__':
    x = np.linspace(0, 6.28 * 2, 32)
    x = np.cos(x)
    xx = np.square(x)
    xx -= np.mean(xx)
    x_f = scipy.fftpack.fft(x)
    xx_f = scipy.fftpack.fft(xx)

    plt.plot(np.abs(x_f[1:len(x)//2]), linewidth=2)
    plt.plot(np.abs(xx_f[1:len(x)//2]), linewidth=2)

    # expected_output = np.concatenate([np.expand_dims(np.real(x_f), -1), np.expand_dims(np.imag(x_f), -1)], axis=1)
    expected_output = np.abs(x_f)
    trained_model = train_fft_model(x, expected_output)

    trained_output_x = trained_model(torch.tensor(x)).cpu().detach().numpy()[0]
    trained_output_xx = trained_model(torch.tensor(xx)).cpu().detach().numpy()[0]
    trained_output_x = np.sqrt(np.sum(np.square(trained_output_x), axis=1))
    trained_output_xx = np.sqrt(np.sum(np.square(trained_output_xx), axis=1))

    plt.plot(np.abs(trained_output_x[1:len(x)//2]), linewidth=1)
    plt.plot(np.abs(trained_output_xx[1:len(x)//2]), linewidth=1)
    plt.show()
