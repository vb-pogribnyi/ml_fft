import numpy as np
import ml_fft
import scipy.fftpack

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

signal_len = 8
n_freq = 12
n_amp = 128

def generate_sine(freq, amp):
	result = np.linspace(0, 6.28*freq, signal_len)
	result = np.sin(result) * amp

	label = np.zeros(n_freq)
	label[freq] = amp

	return result, label

def generate_signal(n_signals):
	wave = np.zeros(signal_len)
	label = np.zeros(n_freq)
	for i in range(n_signals):
		freq = np.random.randint(1, n_freq)
		amp = np.random.randint(50, n_amp) / 100
		single_wave, single_label = generate_sine(freq, amp)
		wave += single_wave
		wave += np.random.rand(len(single_wave)) * 0.25
		label += single_label

		# print(freq, amp)

	return wave, label

while True:
	sig, _ = generate_signal(3)
	plt.plot(sig)
	plt.savefig('figures/sample.png')
	y_f = scipy.fftpack.fft(sig)
	y = np.concatenate([y_f.real.reshape(1, -1, 1), y_f.imag.reshape(1, -1, 1)], axis=2)
	loss, weights = ml_fft.train_fft_model(sig, y)
	if loss > 0.001 and loss < 0.01:
		weights = weights.detach()
		np.save('weights/weights.npy', weights)
		np.save('weights/x.npy', sig)
		np.save('weights/y.npy', y)
		print('Found!')
		break

exit()

'''
x = np.load('x.npy')
y = np.load('y.npy')

print(x.shape, y.shape)
weights = ml_fft.train_fft_model(x, y)

print(weights)
'''
