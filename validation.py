import shutil
import time
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
from IPython.core.debugger import Pdb
# from scheduler import CustomReduceLROnPlateau
from vqa import VQAModel
import json
from dataloader import VQADataset, VQABatchSampler
from torch.autograd import Variable
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, criterion, optimizer, use_gpu=False):

    model.train()
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    step = 0

    for questions, images, image_ids, answers, ques_ids in dataloader:

        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(), images.cuda(), image_ids.cuda(), answers.cuda()
        
        questions, images, answers = Variable(questions).transpose(0, 1), Variable(images), Variable(answers)

        optimizer.zero_grad()
        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)
        loss = criterion(ans_scores, answers)

        loss.backward()
        optimizer.step()

        running_loss += loss.data
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)
        step += 1
        if step % 5000 == 0:
            print('running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
                running_loss / example_count, running_corrects, example_count, (float(running_corrects) / example_count) * 100))

    loss = running_loss / example_count
    acc = (running_corrects / len(dataloader.dataset)) * 100
    print('Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                           acc, running_corrects, example_count))
    return model, loss, acc

def main():
    start_epoch = 0 
    num_epochs = 10

    model = VQAModel(extract_img_features = False)
    data_dir = './data'
    img_dir = 'val2014'
    save_dir = 'tb'
    use_gpu = False

    os.makedirs(save_dir, exist_ok = True)

    train_dataset = VQADataset(data_dir = './data', img_dir = img_dir, phase = 'train', raw_images = True) 
    train_sampler = VQABatchSampler(train_dataset, 2)

    train_dataloader = DataLoader(train_dataset, batch_sampler = train_sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    best_acc = 0

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)
        train_begin = time.time()
        model, train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, use_gpu)

        print("Training Loss and Accuracy", train_loss, train_acc)


if __name__ == "__main__":
    main()


