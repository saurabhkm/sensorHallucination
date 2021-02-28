def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from visdom import Visdom
import numpy as np
import os, shutil
import torch
import pickle
from collections import OrderedDict
from sklearn.metrics import confusion_matrix

class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', modality=''):
        self.viz = Visdom()
        self.env = env_name
        self.modality = modality
        self.plots = {}
        self.paramList = {}
    def argsTile(self, argsDict):
        self.paramList = self.viz.text(self.modality+'\n <b>Training Parameters:</b>\n', env=self.env, opts=dict(width=220,height=320))
        for key, value in argsDict.items():
            self.viz.text(str(key) + ' = ' + str(value) + '\n', env=self.env, win=self.paramList, append=True)
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env,
                opts=dict(legend=[split_name], title=self.modality+var_name, xlabel='Epochs', ylabel=var_name))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')
    def showImage(self, imageTensor):
        # self.viz.image(imageTensor, win=self.images, env=self.env, opts=dict(title='Original and Reconstructed', caption='How random.'),)
        self.viz.images(imageTensor, win=self.images, env=self.env, opts=dict(title='Original and Reconstructed', caption='How random.', nrow=2),)
    def plotPerformance(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=x, Y=y, env=self.env,
                opts=dict(legend=[split_name], title=var_name, xlabel='Epochs', ylabel=var_name))
        else:
            self.viz.line(X=x, Y=y, env=self.env, win=self.plots[var_name], name=split_name, update='append')

class diskWriter(object):
    """Writes to CSV"""
    def __init__(self, directory):
        root = 'runs'
        if os.path.exists(os.path.join(root, directory)):
            shutil.rmtree(os.path.join(root, directory))
        self.performanceFilename = os.path.join(root, directory, "performance.csv")
        self.parametersFilename = os.path.join(root, directory, "parameters.csv")
        self.mediaFile = os.path.join(root, directory, "media.pkl")
        os.makedirs(os.path.join(root, directory))
        open(self.performanceFilename, 'a').close()
        open(self.parametersFilename, 'a').close()
    def writePerformance(self, datalist):
        with open(self.performanceFilename, "a") as file:
            file.write(','.join(map(str, datalist)))
            file.write('\n')
    def writeParameters(self, paramsDict):
        with open(self.parametersFilename, "a") as file:
            for key, value in paramsDict.items():
                file.write(str(key) + ', ' + str(value) + '\n')
    def writeMedia(self, mediaArray):
        pickle.dump(mediaArray, open(self.mediaFile, "wb" ))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(name, filename):
	modelDir = "/home/SharedData/saurabh/HALL/models/%s/"%(name)
	checkpoint = torch.load(modelDir + filename)
	stateDict = checkpoint['stateDict']
	return stateDict

def save_model(state, name, filename):
	saveDir = "/home/SharedData/saurabh/HALL/models/%s/"%(name)
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	torch.save(state, saveDir + filename)

# Model Utils
def appendToKeys(stateDict, val):
    newDict = OrderedDict()
    for k in stateDict:
        newDict[k[:2]+val+k[2:]] = stateDict[k]
    return newDict

def appendToKeysNew(stateDict, stream):
    newDict = OrderedDict()
    for k in stateDict:
        newDict[stream+'.'+k] = stateDict[k]
    return newDict

def getClasswiseAccuracy(labels, predicted):
    bins = np.zeros(len(np.unique(labels))+1)
    bins[-1] = 8
    bins[:-1] = np.unique(labels)
    classCounts, _ = np.histogram(labels, bins)
    classWiseAccs = np.divide(np.diagonal(confusion_matrix(labels, predicted)), classCounts)*100
    return classWiseAccs