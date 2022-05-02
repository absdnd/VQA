import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from IPython.core.debugger import Pdb

from preprocess import preprocess
from dataloader import VQADataset, VQABatchSampler
from train import train_model, test_model
import datetime
import shutil
from vqa import VQAModel


def load_datasets(config, phases):
	config = config['data']
	if 'preprocess' in config and config['preprocess']:
		print('Preprocessing datasets')
		preprocess(
			data_dir=config['dir'],
			train_ques_file=config['train']['ques'],
			train_ans_file=config['train']['ans'],
			val_ques_file=config['val']['ques'],
			val_ans_file=config['val']['ans'])

	print('Loading preprocessed datasets')
	datafiles = {x: '{}_small.pkl'.format(x) for x in phases}
	raw_images = 'preprocess' in config['images'] and config['images']['preprocess']
	
	if raw_images:
		img_dir = {x: config[x]['img_dir'] for x in phases}
	else:
		img_dir = {x: config[x]['emb_dir'] for x in phases}

	datasets = {x: VQADataset(data_dir=config['dir'], qafile=datafiles[x], img_dir=img_dir[x], phase=x,
							  img_scale=config['images']['scale'], img_crop=config['images']['crop'], raw_images=raw_images) for x in phases}
	batch_samplers = {x: VQABatchSampler(
		datasets[x], config[x]['batch_size']) for x in phases}

	dataloaders = {x: DataLoader(
		datasets[x], batch_sampler=batch_samplers[x], num_workers=config['loader']['workers']) for x in phases}
	dataset_sizes = {x: len(datasets[x]) for x in phases}
	print(dataset_sizes)
	print("ques vocab size: {}".format(len(VQADataset.ques_vocab)))
	print("ans vocab size: {}".format(len(VQADataset.ans_vocab)))
	return dataloaders, VQADataset.ques_vocab, VQADataset.ans_vocab


########### Main function with train and validation ##################
def main(config):
	if config['mode'] == 'test':
		phases = ['train', 'test']
	else: 
		phases = ['train', 'val']

	######### Dataloader, ques_vocab and answer vocabulary. ############
	dataloaders, ques_vocab, ans_vocab = load_datasets(config, phases)

	if config['model']['type'] == 'vqa':
		model = VQAModel(mode=config['mode'], **config['model']['params'])

	elif config['model']['type'] == 'attn':
		model = AttnModel(mode=config['mode'], **config['model']['params'])

	criterion = nn.CrossEntropyLoss()
	save_dir = config['save_dir']
	best_acc = 0
	startEpoch = 0

	if config['optim']['class'] == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
							  **config['optim']['params'])
	elif config['optim']['class'] == 'rmsprop':
		optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
								  **config['optim']['params'])
	else:
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
							   **config['optim']['params'])

	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	if config['mode'] == 'train':
		opt_model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, save_dir,
							num_epochs=config['optim']['n_epochs'], use_gpu=config['use_gpu']) #, best_accuracy=best_acc, start_epoch=startEpoch)

	elif config['mode'] == 'test': 
		test_model(model, dataloaders['test'], VQADataset.ans_vocab, 'log.out', use_gpu=config['use_gpu'])



if __name__ == '__main__': 

	LOAD_PATH = 'configs/config_vanilla.yaml'
	config = yaml.load(open(LOAD_PATH))
	config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()

	run_folder = os.path.join('runs', config['model']['type'])
	os.makedirs(run_folder, exist_ok = True)

	curTime = datetime.datetime.now()
	curTimeFolder = os.path.join(run_folder, '{}:{}:{}'.format(curTime.hour, curTime.minute, curTime.second))
	os.makedirs(curTimeFolder, exist_ok = True)
	config['save_dir'] = curTimeFolder

	shutil.copy(LOAD_PATH, curTimeFolder)

	torch.manual_seed(config['seed'])
	torch.cuda.manual_seed(config['seed'])
	main(config)