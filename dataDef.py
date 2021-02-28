import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

# Ready the training dataset
class genericDataset(Dataset):
	panPath = '/home/SharedData/saurabh/HS/panchromaticData.npy'
	msPath = '/home/SharedData/saurabh/HS/multispectralData.npy'
	panmsLabelsPath = '/home/SharedData/saurabh/HS/labelsData.npy'
	hsPath = '/home/SharedData/saurabh/HS/paviaUForegroundSpectral.npy'
	hsLabelsPath = '/home/SharedData/saurabh/HS/paviaUForegroundLabels.npy'
	def __init__(self, modality):
		if modality == 'PAN':
			self.data = np.load(self.panPath)
			self.labels = np.load(self.panmsLabelsPath)
		elif modality == 'MS':
			self.data = np.load(self.msPath)
			self.labels = np.load(self.panmsLabelsPath)
		elif modality == 'HS1':
			self.data = np.load(self.hsPath)[:,51:]
			self.labels = np.load(self.hsLabelsPath)
		elif modality == 'HS2':
			self.data = np.load(self.hsPath)[:,:51]
			self.labels = np.load(self.hsLabelsPath)
		# self.data = torch.Tensor(self.data)
		# self.labels = torch.squeeze(torch.LongTensor(self.labels - 1))

	def __getitem__(self, idx):
		sample = torch.Tensor(self.data[idx])
		label = torch.squeeze(torch.LongTensor(self.labels[:,idx] - 1))
		return sample, label
		# return self.data[idx], self.labels[idx]

	def __len__(self):
		return self.data.shape[0]

class twoStreamDataset(Dataset):
	panPath = '/home/SharedData/saurabh/HS/panchromaticData.npy'
	msPath = '/home/SharedData/saurabh/HS/multispectralData.npy'
	panmsLabelsPath = '/home/SharedData/saurabh/HS/labelsData.npy'
	hsPath = '/home/SharedData/saurabh/HS/paviaUForegroundSpectral.npy'
	hsLabelsPath = '/home/SharedData/saurabh/HS/paviaUForegroundLabels.npy'
	def __init__(self, stream1, stream2):
		if stream1 == 'PAN' and stream2 == 'MS':
			self.data1 = np.load(self.panPath)
			self.data2 = np.load(self.msPath)
			self.labels = np.load(self.panmsLabelsPath)
		elif stream1 == 'PAN' and stream2 == 'PAN':
			self.data1 = np.load(self.panPath)
			self.data2 = np.load(self.panPath)
			self.labels = np.load(self.panmsLabelsPath)
		elif stream1 == 'MS' and stream2 == 'MS':
			self.data1 = np.load(self.msPath)
			self.data2 = np.load(self.msPath)
			self.labels = np.load(self.panmsLabelsPath)
		elif stream1 == 'MS' and stream2 == 'PAN':
			self.data1 = np.load(self.msPath)
			self.data2 = np.load(self.panPath)
			self.labels = np.load(self.panmsLabelsPath)
		elif stream1 == 'HS1' and stream2 == 'HS2':
			self.data1 = np.load(self.hsPath)[:,51:]
			self.data2 = np.load(self.hsPath)[:,:51]
			self.labels = np.load(self.hsLabelsPath)
		elif stream1 == 'HS1' and stream2 == 'HS1':
			self.data1 = np.load(self.hsPath)[:,51:]
			self.data2 = np.load(self.hsPath)[:,51:]
			self.labels = np.load(self.hsLabelsPath)
		elif stream1 == 'HS2' and stream2 == 'HS2':
			self.data1 = np.load(self.hsPath)[:,:51]
			self.data2 = np.load(self.hsPath)[:,:51]
			self.labels = np.load(self.hsLabelsPath)
		elif stream1 == 'HS2' and stream2 == 'HS1':
			self.data1 = np.load(self.hsPath)[:,:51]
			self.data2 = np.load(self.hsPath)[:,51:]
			self.labels = np.load(self.hsLabelsPath)
		# self.data1 = torch.Tensor(self.data1)
		# self.data2 = torch.Tensor(self.data2)
		# self.labels = torch.squeeze(torch.LongTensor(self.labels - 1))

	def __getitem__(self, idx):
		sample1 = torch.Tensor(self.data1[idx])
		sample2 = torch.Tensor(self.data2[idx])
		label = torch.squeeze(torch.LongTensor(self.labels[:,idx] - 1))
		return sample1, sample2, label
		# return self.data1[idx], self.data2[idx], self.labels[idx]

	def __len__(self):
		return self.data1.shape[0]

def testTrainSplit(numSamples, splitRatio):
	''' Get train and validation sampler as per 'splitRatio' and total sample count 'numSamples'''
	indices = list(range(numSamples))
	split = int(splitRatio * numSamples)

	validation_idx = np.random.choice(indices, size=split, replace=False)
	train_idx = list(set(indices) - set(validation_idx))

	trainSampler = SubsetRandomSampler(train_idx)
	validationSampler = SubsetRandomSampler(validation_idx)

	return trainSampler, validationSampler