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
from tensorboardX import SummaryWriter
import yaml



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


    ### Loss and Accuracy obtained by overfitting a few examples #### 

    loss = running_loss / example_count
    acc = running_corrects/ example_count
    # acc = (running_corrects / len(dataloader.dataset)) * 100
    print('Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                           acc, running_corrects, example_count))
    return model, loss, acc

### Validate the model after training on it #### 
def validate(model, epoch, dataloader, criterion, use_gpu=False):
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    example_count = 0

    # import pdb
    # pdb.set_trace()

    for questions, images, image_ids, answers, ques_ids in dataloader:

        # if epoch == 8:
            # import pdb
            # pdb.set_trace()

        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(
            ), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(
            0, 1), Variable(images), Variable(answers)

        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)
        loss = criterion(ans_scores, answers)

        running_loss += loss.data
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)

    loss = running_loss / example_count
    acc = running_corrects/ example_count
    # acc = (running_corrects / len(dataloader.dataset)) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                                acc, running_corrects, example_count))
    return loss, acc

def test_model(model, dataloader, itoa, outputfile, use_gpu=False):
    
    model.eval()
    example_count = 0
    test_begin = time.time()
    outputs = []

    for questions, images, image_ids, answers, ques_ids in dataloader:

        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(
            ), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(
            0, 1), Variable(images), Variable(answers)

        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)

        outputs.extend([{'question_id': ques_ids[i].item(), 'answer': itoa[str(
            preds.data[i].item())]} for i in range(ques_ids.size(0))])

        if example_count % 100 == 0:
            print('(Example Count: {})'.format(example_count))

        example_count += answers.size(0)

    json.dump(outputs, open(outputfile, 'w'))
    print('(Example Count: {})'.format(example_count))
    test_time = time.time() - test_begin
    print('Test Time: {:.0f}m {:.0f}s'.format(test_time // 60, test_time % 60))


def train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, save_dir, num_epochs, use_gpu, CKPT_FREQ = 10):

    ckpt_dir = os.path.join(save_dir, 'ckpts')
    tb_dir = os.path.join(save_dir, 'tb')
    os.makedirs(ckpt_dir, exist_ok = True)
    os.makedirs(tb_dir, exist_ok = True)

    writer = SummaryWriter(tb_dir)
    best_acc = 0

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_begin = time.time()

        model, train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, use_gpu)
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_acc, epoch)

        val_loss, val_acc = validate(model, epoch, dataloaders['val'], criterion, use_gpu)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)

        is_best = val_acc > best_acc
        if is_best: 
            best_acc = val_acc 
            best_model_wts = model.state_dict()

        if epoch % CKPT_FREQ == 0:
            save_checkpoint(ckpt_dir, {
                'epoch': epoch,
                'best_acc': best_acc,
                'state_dict': model.state_dict(),
            }, is_best, epoch)

        writer.export_scalars_to_json(save_dir + "/all_scalars.json")

def save_checkpoint(save_dir, state, is_best, epoch):
    savepath = save_dir + '/' + 'ckpt.{}.pth'.format(epoch)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, save_dir + '/' + 'model_best.pth.tar')


def train_model_small():


    start_epoch =  0
    num_epochs = 30

    model = VQAModel(extract_img_features = False)
    data_dir = './data'
    img_dir = 'val2014'
    save_dir = 'tb'
    use_gpu = False

    os.makedirs(save_dir, exist_ok = True)

    train_dataset = VQADataset(data_dir = './data', qafile = 'train_small.pkl' , img_dir = img_dir, phase = 'train', raw_images = True) 
    validation_dataset = VQADataset(data_dir = './data', qafile = 'val_small.pkl', img_dir = img_dir, phase = 'val', raw_images = True)

    train_sampler = VQABatchSampler(train_dataset, 2)
    train_dataloader = DataLoader(train_dataset, batch_sampler = train_sampler)

    val_sampler = VQABatchSampler(validation_dataset, 2)
    val_dataloader = DataLoader(validation_dataset, batch_sampler = val_sampler)
    writer = SummaryWriter(save_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    best_acc = 0

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_begin = time.time()

        model, train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, use_gpu)
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_acc, epoch)

        val_loss, val_acc = validate(model, epoch, val_dataloader, criterion, use_gpu)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)

        is_best = val_acc > best_acc
        if is_best: 
            best_acc = val_acc
            best_model_wts = model.state_dict()

        writer.export_scalars_to_json(save_dir + "/all_scalars.json")




    # start_epoch =  0
    # num_epochs = 30

    # model = VQAModel(extract_img_features = False)
    # data_dir = './data'
    # img_dir = 'val2014'
    # save_dir = 'tb'
    # use_gpu = False

    # os.makedirs(save_dir, exist_ok = True)

    # train_dataset = VQADataset(data_dir = './data', img_dir = img_dir, phase = 'train', raw_images = True) 
    # validation_dataset = VQADataset(data_dir = './data', img_dir = img_dir, phase = 'val', raw_images = True)

    # train_sampler = VQABatchSampler(train_dataset, 2)
    # train_dataloader = DataLoader(train_dataset, batch_sampler = train_sampler)

    # val_sampler = VQABatchSampler(validation_dataset, 2)
    # val_dataloader = DataLoader(validation_dataset, batch_sampler = val_sampler)
    # writer = SummaryWriter(save_dir)
    # # criterion = nn.CrossEntropyLoss()
    # # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # # best_acc = 0

    # for epoch in range(0, num_epochs):
    #     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #     print('-' * 10)
    #     train_begin = time.time()

    #     model, train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, use_gpu)
    #     writer.add_scalar('Train Loss', train_loss, epoch)
    #     writer.add_scalar('Train Accuracy', train_acc, epoch)

    #     val_loss, val_acc = validate(model, epoch, val_dataloaders['val'], criterion, use_gpu)
    #     writer.add_scalar('Validation Loss', val_loss, epoch)
    #     writer.add_scalar('Validation Accuracy', val_acc, epoch)

    #     is_best = val_acc > best_acc
    #     if is_best: 
    #         best_acc = val_acc
    #         best_model_wts = model.state_dict()

    #     writer.export_scalars_to_json(save_dir + "/all_scalars.json")






if __name__ == "__main__":
    config = yaml.load('configs/config_vanilla.yaml')
    train_model_small(config)


