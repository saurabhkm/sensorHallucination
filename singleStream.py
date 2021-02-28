import torch, argparse
import numpy as np
import modelDef, dataDef, engine, utils
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-valRatio', type=float, help='Validation data ratio')
parser.add_argument('-learningRate', type=float, help='Learning rate for the optimizer')
parser.add_argument('-batchSize', type=int, help='Mini batch size')
parser.add_argument('-nEpochs', type=int, help='Number of Epochs')
parser.add_argument('-instance', type=int, help='Instance count')
parser.add_argument('-modality', type=str, help='Modality to use')
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
    plotter = utils.VisdomPlotter(env_name=args.name, modality=args.modality)
    plotter.argsTile(args.__dict__)

# Ready the training and validation data
dataset = dataDef.genericDataset(args.modality)
trainSampler, validationSampler = dataDef.testTrainSplit(len(dataset), args.valRatio)
trainLoader = torch.utils.data.DataLoader(dataset, sampler=trainSampler, batch_size=args.batchSize, num_workers=8)
valLoader = torch.utils.data.DataLoader(dataset, sampler=validationSampler, batch_size=args.trainTestBS*args.batchSize, num_workers=8)

# Initialize the network, loss and optimizer
if args.modality == 'PAN':
	model = modelDef.panNet()
elif args.modality == 'MS':
	model = modelDef.msNet()
elif args.modality == 'HS1':
	model = modelDef.HS1(dataset.data.shape[1])
elif args.modality == 'HS2':
	model = modelDef.HS2(dataset.data.shape[1])

model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate)

# Training and Inference
print('Training (' + args.modality + ') Network...')
bestAcc = 0
for epoch in tqdm(range(args.nEpochs)):
    engine.train(model, trainLoader, optimizer, criterion, device, plotter, epoch)
    accuracy, classWiseAcc = engine.validate(model, valLoader, criterion, device, plotter, epoch)
    if accuracy > bestAcc:
        utils.save_model({'stateDict': model.state_dict()}, args.name, args.modality+'net_instance_'+str(args.instance)+'.pth.tar')
        bestAcc = accuracy
print('Best Accuracy: '+str(bestAcc))
# print('Classwise Accuracy: '+str(classWiseAcc))