import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

def train(model, trainLoader, optimizer, criterion, device, plotter, epoch):
	losses = utils.AverageMeter()
	accuracies = utils.AverageMeter()
	model.train()
	running_loss = 0.0
	for i, (inputs, labels) in enumerate(trainLoader, 0):
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# running_loss += loss.item()
		losses.update(loss.item(), labels.size(0))
	if plotter != None:
		plotter.plot('Loss', 'train', epoch, losses.avg)
		plotter.plot('Accuracy', 'train', epoch, accuracies.avg)

def validate(model, testLoader, criterion, device, plotter, epoch):
	losses = utils.AverageMeter()
	accuracies = utils.AverageMeter()
	correct = 0
	total = 0
	# prevTotal = 0
	# labelsHolder = np.zeros((40000,))
	# predictedHolder = np.zeros_like(labelsHolder)
	model.eval()
	with torch.no_grad():
		running_loss = 0.0
		for (inputs, labels) in testLoader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			# running_loss += loss.item()
			# labelsHolder[prevTotal:total] = labels.cpu().numpy()
			# predictedHolder[prevTotal:total] = predicted.cpu().numpy()
			# prevTotal = total
			losses.update(loss.item(), labels.size(0))
		accuracy = (100*correct)/total
		accuracies.update(accuracy, len(testLoader))
	if plotter != None:
		plotter.plot('Loss', 'test', epoch, losses.avg)
		plotter.plot('Accuracy', 'test', epoch, accuracies.avg)
	# print(classification_report(labelsHolder, predictedHolder))
	# classWiseAcc = utils.getClasswiseAccuracy(labelsHolder, predictedHolder)
	return accuracy, 0 #, classWiseAcc

def twoStreamTrain(model, trainLoader, optimizer, criterion, device, plotter, epoch):
	losses = utils.AverageMeter()
	accuracies = utils.AverageMeter()
	model.train()
	running_loss = 0.0
	for i, (inputs1, inputs2, labels) in enumerate(trainLoader, 0):
		inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
		outputs = model(inputs1, inputs2)
		loss = criterion(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# running_loss += loss.item()
		losses.update(loss.item(), labels.size(0))
	if plotter != None:
		plotter.plot('Loss', 'train', epoch, losses.avg)
		plotter.plot('Accuracy', 'train', epoch, accuracies.avg)

def twoStreamValidate(model, testLoader, criterion, device, plotter, epoch):
	losses = utils.AverageMeter()
	accuracies = utils.AverageMeter()
	correct = 0
	total = 0
	# prevTotal = 0
	# labelsHolder = np.zeros((40000,))
	# predictedHolder = np.zeros_like(labelsHolder)
	model.eval()
	with torch.no_grad():
		running_loss = 0.0
		for (inputs1, inputs2, labels) in testLoader:
			inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
			outputs = model(inputs1, inputs2)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			# running_loss += loss.item()
			# labelsHolder[prevTotal:total] = labels.cpu().numpy()
			# predictedHolder[prevTotal:total] = predicted.cpu().numpy()
			# prevTotal = total
			losses.update(loss.item(), labels.size(0))
		accuracy = (100*correct)/total
		accuracies.update(accuracy, len(testLoader))
	if plotter != None:
		plotter.plot('Loss', 'test', epoch, losses.avg)
		plotter.plot('Accuracy', 'test', epoch, accuracies.avg)
	# print(classification_report(labelsHolder, predictedHolder))
	# classWiseAcc = utils.getClasswiseAccuracy(labelsHolder, predictedHolder)
	return accuracy, 0 #, classWiseAcc

def hallucinationTrain(model, trainLoader, optimizer, device, T, Lambda, alpha, plotter, epoch):
	losses = utils.AverageMeter()
	accuracies = utils.AverageMeter()
	model.train()
	running_loss = 0.0
	for i, (inputs1, inputs2, labels) in enumerate(trainLoader, 0):
		inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
		outputs, teacher_outputs = model(inputs1, inputs2)
		# loss1 = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), #fullFlow7
		loss1 = nn.KLDivLoss()(F.softmax(outputs/T, dim=1),	# Gives better hallucination net than above loss1
			F.softmax(teacher_outputs/T, dim=1)) * (Lambda * T * T) + F.cross_entropy(outputs, labels) * (1. - Lambda)
		# loss1 = nn.KLDivLoss()(F.softmax(outputs/T, dim=1),
		# 	F.softmax(teacher_outputs/T, dim=1)) * (Lambda) + F.cross_entropy(outputs, labels) * (1. - Lambda)
		loss2 = nn.MSELoss()(F.softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1))
		loss = loss1*alpha + (1-alpha)*loss2
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# running_loss += loss.item()
		losses.update(loss.item(), labels.size(0))
	if plotter != None:
		plotter.plot('Loss', 'train', epoch, losses.avg)
		plotter.plot('Accuracy', 'train', epoch, accuracies.avg)

def hallucinationValidate(model, testLoader, criterion, device, plotter, epoch):
	losses = utils.AverageMeter()
	accuracies = utils.AverageMeter()
	correct = 0
	total = 0
	# prevTotal = 0
	# labelsHolder = np.zeros((40000,))
	# predictedHolder = np.zeros_like(labelsHolder)
	model.eval()
	with torch.no_grad():
		running_loss = 0.0
		for (inputs1, _, labels) in testLoader:
			inputs1, labels = inputs1.to(device), labels.to(device)
			outputs = model(inputs1)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			# running_loss += loss.item()
			# labelsHolder[prevTotal:total] = labels.cpu().numpy()
			# predictedHolder[prevTotal:total] = predicted.cpu().numpy()
			# prevTotal = total
			losses.update(loss.item(), labels.size(0))
		accuracy = (100*correct)/total
		accuracies.update(accuracy, len(testLoader))
	if plotter != None:
		plotter.plot('Loss', 'test', epoch, losses.avg)
		plotter.plot('Accuracy', 'test', epoch, accuracies.avg)
	# print(classification_report(labelsHolder, predictedHolder))
	# classWiseAcc = utils.getClasswiseAccuracy(labelsHolder, predictedHolder)
	return accuracy, 0 #, classWiseAcc