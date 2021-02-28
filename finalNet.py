import torch, argparse
import numpy as np
import modelDef, dataDef, engine, utils
from collections import OrderedDict
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-valRatio', type=float, help='Validation data ratio')
parser.add_argument('-learningRate', type=float, help='Learning rate for the optimizer')
parser.add_argument('-batchSize', type=int, help='Mini batch size')
parser.add_argument('-nEpochs', type=int, help='Number of Epochs')
parser.add_argument('-instance', type=int, help='Instance count')
parser.add_argument('-modality1', type=str, help='Modality1 to use')
parser.add_argument('-modality2', type=str, help='Modality2 to use')
parser.add_argument('-name', default='default', type=str, help='name of experiment')
parser.add_argument('-plot', action="store_true", default=False, help='Enable plotting')
parser.add_argument('-trainTestBS', type=int, help='Ratio of Train BS to test BS')

args = parser.parse_args()
device = torch.device('cuda')

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)
torch.backends.cudnn.benchmark = True

plotter = None
if args.plot == True:
    plotter = utils.VisdomPlotter(env_name=args.name, modality='finalNet')
    plotter.argsTile(args.__dict__)

# Ready the training and validation data
dataset = dataDef.twoStreamDataset(args.modality1, args.modality1)
trainSampler, validationSampler = dataDef.testTrainSplit(len(dataset), args.valRatio)
trainLoader = torch.utils.data.DataLoader(dataset, sampler=trainSampler, batch_size=args.batchSize, num_workers=8)
valLoader = torch.utils.data.DataLoader(dataset, sampler=validationSampler, batch_size=args.trainTestBS*args.batchSize, num_workers=8)

# Initialize our network
if args.modality1 == 'PAN':
    net1 = modelDef.panNet()
    net2 = modelDef.panNet()
    model = modelDef.twoStreamNet(net1, net2)
elif args.modality1 == 'MS':
    net1 = modelDef.msNet()
    net2 = modelDef.msNet()
    model = modelDef.twoStreamNet(net1, net2)
elif args.modality1 == 'HS1':
    net1 = modelDef.HS1(dataset.data1.shape[1])
    net2 = modelDef.HS1(dataset.data1.shape[1])
    model = modelDef.twoStreamNetHS(net1, net2)
elif args.modality1 == 'HS2':
    net1 = modelDef.HS2(dataset.data1.shape[1])
    net2 = modelDef.HS2(dataset.data1.shape[1])
    model = modelDef.twoStreamNetHS(net1, net2)

twoStreamDict = utils.load_model(args.name, 'twoStreamNet_'+args.modality1+'-'+args.modality2+'_instance_'+str(args.instance)+'.pth.tar')
hallucinationDict = utils.load_model(args.name, 'hallucinationNet_'+args.modality2+'_instance_'+str(args.instance)+'.pth.tar')

# Initialize with learned streams
newDict = OrderedDict()
for k in model.state_dict():
    # if k[:6]=='stream1' or k[:3]=='fc.':
    if k[:7]=='stream1' or k[:4]=='fuse':
        newDict[k] = twoStreamDict[k]
for k in model.state_dict():
    if k[:7]=='stream2':
        newDict['stream2.'+k[8:]] = hallucinationDict['stream1.'+k[8:]]

state = model.state_dict()
state.update(newDict)
model.load_state_dict(state)

for n, p in model.named_parameters():
    if n[:7]=='stream2':
        p.requires_grad = False

model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learningRate)

# Training and Inference
print('Training Final Two Stream ('+ args.modality1+'-'+args.modality1 +') Network...')
bestAcc = 0
for epoch in tqdm(range(args.nEpochs)):
    engine.twoStreamTrain(model, trainLoader, optimizer, criterion, device, plotter, epoch)
    accuracy, classWiseAcc = engine.twoStreamValidate(model, valLoader, criterion, device, plotter, epoch)
    if accuracy > bestAcc:
        bestAcc = accuracy
print('Best Accuracy: '+str(bestAcc))
# print('Classwise Accuracy: '+str(classWiseAcc))