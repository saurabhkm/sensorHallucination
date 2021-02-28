import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

class panNet(nn.Module):
	def __init__(self):
		super(panNet, self).__init__()
		self.cv1 = nn.Conv2d(1, 4, 5)
		self.cv2 = nn.Conv2d(4, 16, 5)
		self.cv3 = nn.Conv2d(16, 32, 5)
		self.cv4 = nn.Conv2d(32, 64, 5)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.pool3 = nn.MaxPool2d(3, 3)
		self.fc1 = nn.Linear(64 * 7 * 7, 448)
		self.fc2 = nn.Linear(448, 120)
		self.fc3 = nn.Linear(120, 84)
		self.fc4 = nn.Linear(84, 8)
	def forward(self, x):
		x = self.pool3(F.relu(self.cv1(x)))
		x = self.pool2(F.relu(self.cv2(x)))
		x = self.pool2(F.relu(self.cv3(x)))
		x = self.pool2(F.relu(self.cv4(x)))
		x = x.view(-1, 64 * 7 * 7)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

class msNet(nn.Module):
	def __init__(self):
		super(msNet, self).__init__()
		self.cv1 = nn.Conv2d(4, 8, 5)
		self.cv2 = nn.Conv2d(8, 16, 5)
		self.cv3 = nn.Conv2d(16, 32, 5)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.pool3 = nn.MaxPool2d(3, 3)
		self.fc1 = nn.Linear(32 * 3 * 3, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 8)
	def forward(self, x):
		x = self.pool2(F.relu(self.cv1(x)))
		x = self.pool2(F.relu(self.cv2(x)))
		x = self.pool3(F.relu(self.cv3(x)))
		x = x.view(-1, 32 * 3 * 3)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class twoStreamNet(nn.Module):
	def __init__(self, stream1, stream2):
		super(twoStreamNet, self).__init__()
		self.stream1 = stream1
		self.stream2 = stream2
		# Fusion layer
		# self.fuse = nn.Linear(8, 8)
		self.fuse = nn.Linear(16, 8)
	def forward(self, x1, x2):
		x1 = F.softmax(self.stream1(x1))
		x2 = F.softmax(self.stream2(x2))
		# fused = torch.div(torch.add(x1, x2), 2)
		fused = torch.cat((x1, x2), 1)
		out = self.fuse(fused)
		return out

class halucinationNet(nn.Module):
	def __init__(self, stream1, stream2):
		super(halucinationNet, self).__init__()
		self.stream1 = stream1
		self.stream2 = stream2
	def forward(self, x1, x2):
		x1 = self.stream1(x1)
		x2 = self.stream2(x2)
		return x1, x2

# Changing network depth-----------------
class panNetSmall(nn.Module):
	def __init__(self):
		super(panNetSmall, self).__init__()
		self.cv1 = nn.Conv2d(1, 4, 7)
		self.cv2 = nn.Conv2d(4, 16, 5)
		self.cv3 = nn.Conv2d(16, 32, 3)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.pool3 = nn.MaxPool2d(3, 3)
		self.pool5 = nn.MaxPool2d(5, 5)
		self.fc1 = nn.Linear(32 * 7 * 7, 224)
		self.fc2 = nn.Linear(224, 60)
		self.fc3 = nn.Linear(60, 8)
	def forward(self, x):
		x = self.pool5(F.relu(self.cv1(x)))
		x = self.pool2(F.relu(self.cv2(x)))
		x = self.pool3(F.relu(self.cv3(x)))
		x = x.view(-1, 32 * 7 * 7)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class msNetSmall(nn.Module):
	def __init__(self):
		super(msNetSmall, self).__init__()
		self.cv1 = nn.Conv2d(4, 8, 7)
		self.cv2 = nn.Conv2d(8, 16, 3)
		self.pool3 = nn.MaxPool2d(3, 3)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16 * 9 * 9, 100)
		self.fc2 = nn.Linear(100, 8)
	def forward(self, x):
		x = self.pool2(F.relu(self.cv1(x)))
		x = self.pool3(F.relu(self.cv2(x)))
		x = x.view(-1, 16 * 9 * 9)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

#### HS models---------------------------------------
class HS1(nn.Module):
	def __init__(self, inDims):
		super(HS1, self).__init__()
		self.inDims = inDims
		self.fc1 = nn.Linear(inDims, 36)
		self.fc2 = nn.Linear(36, 18)
		self.fc3 = nn.Linear(18, 9)
	def forward(self, x):
		x = x.view(-1, self.inDims)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class HS2(nn.Module):
	def __init__(self, inDims):
		super(HS2, self).__init__()
		self.inDims = inDims
		self.fc1 = nn.Linear(inDims, 36)
		self.fc2 = nn.Linear(36, 18)
		self.fc3 = nn.Linear(18, 9)
	def forward(self, x):
		x = x.view(-1, self.inDims)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class twoStreamNetHS(nn.Module):
	def __init__(self, stream1, stream2):
		super(twoStreamNetHS, self).__init__()
		self.stream1 = stream1
		self.stream2 = stream2
		# Fusion layer
		# self.fuse = nn.Linear(9, 9)
		self.fuse = nn.Linear(18, 9)
	def forward(self, x1, x2):
		x1 = F.softmax(self.stream1(x1))
		x2 = F.softmax(self.stream2(x2))
		# fused = torch.div(torch.add(x1, x2), 2)
		fused = torch.cat((x1, x2), 1)
		out = self.fuse(fused)
		return out

class halucinationNetHS(nn.Module):
	def __init__(self, stream1, stream2):
		super(halucinationNetHS, self).__init__()
		self.stream1 = stream1
		self.stream2 = stream2
	def forward(self, x1, x2):
		x1 = self.stream1(x1)
		x2 = self.stream2(x2)
		return x1, x2