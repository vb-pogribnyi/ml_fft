import torch
import numpy as np

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
        param_data = torch.rand([self.len // 2, 2])
        self.weight = torch.nn.Parameter(data=param_data, requires_grad=True)

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
    for i in range(50001):
        opt.zero_grad()
        output = fft_model(x)
        err = torch.mean(torch.pow(expected_output - output, 2))
        if i % 100 == 0:
            print(i, err)
        err.backward()
        opt.step()

    return fft_model.weight


# print(model(x))
# exit()
if __name__ == '__main__':
    x = [0., 0.781, 0.975, 0.435, -0.432, -0.974, -0.783, -0.003]
    expected_output = [[
             [-9.9993e-04,  0.0000e+00],
             [ 1.3631e+00, -3.3085e+00],
             [-6.2400e-01,  6.2500e-01],
             [-4.9912e-01,  2.0755e-01],
             [-4.7900e-01,  0.0000e+00],
             [-4.9912e-01, -2.0755e-01],
             [-6.2400e-01, -6.2500e-01],
             [ 1.3631e+00,  3.3085e+00]
    ]]
    trained_weights = train_fft_model(x, expected_output)
    print(trained_weights)
    print('done')
