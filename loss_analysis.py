import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import ml_fft

s1 = [0.391, 1.255]
s2 = [0.747, 0.786]
ss = [0.391, 0.747, 1.255, 0.786]
s1_f = scipy.fftpack.fft(s1)
s2_f = scipy.fftpack.fft(s2)
ss_f = scipy.fftpack.fft(ss)
s1_y = np.concatenate([s1_f.real.reshape(1, -1, 1), s1_f.imag.reshape(1, -1, 1)], axis=2)

model_2 = ml_fft.FFTModel(2)
xs = np.linspace(-1., 1., 10)
ys = np.linspace(0., 2., 10)
losses = np.zeros([len(ys), len(xs)])
for idx_w1, w_1 in enumerate(ys):
    for idx_w2, w_2 in enumerate(xs):
        model_2.weight.data = torch.tensor([[w_1, w_2]])
        output = model_2(torch.tensor(s1))
        err = torch.mean(torch.pow(torch.abs(output - torch.tensor(s1_y)), 2))
        losses[idx_w1, idx_w2] = err.detach().numpy().sum()
print(model_2(torch.tensor(s1)))
plt.contourf(xs, ys, losses, 100)
plt.show()


# func = np.linspace(-1, 1, 51)
# line = np.linspace(-1, 1, 51)
#
# func = np.square(func)
# line = line + 1
#
# print(line[0::10])
# print(func[0::10])
# print((line * func)[0::10])
# print((line - line * func)[0::10])
#
# plt.plot(line)
# plt.plot(func)
# plt.plot(line * func)
# plt.plot(line - line * func)
# plt.show()

# print(s1, ss)
