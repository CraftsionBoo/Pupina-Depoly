import argparse
import torch
import torch.nn.functional as F 
import torch.optim as optim 
import os 
import time
import shutil

# self .py
import model
import dataset 
import utils

# cmd sample : python3 train.py --logdir ./logs --data_root ./pytorch/public_data 
# cmd2 : python3 train.py --logdir ./logs --data_root ./datasets/ --epochs 20 --decreasing_lr "5,10" --log_interval 5 --test_interval 5 --onnx_root ./logs

parser = argparse.ArgumentParser(description="Pytorch Mnist for Alexnet")
parser.add_argument("--wd", type=float, default=0.0001, help="weights decay")
parser.add_argument("--batch_size", type=int, default=64, help="nput batch size for training(default: 64)")
parser.add_argument("--epochs", type=int, default=21, help="umber of epochs to train(default : 1000)")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default : 1e-3)")
parser.add_argument("--gpu", type=None, help="index of gpus to use")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpus to use")
parser.add_argument('--log_interval', type=int, default=20,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=3,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument("--data_root", default="/tmp/public_data/pytorch/", help="folder to save the model")
parser.add_argument("--decreasing_lr", default="12, 15", help="decreasing strategy")
parser.add_argument("--onnx_root", default=None, help="model convert to onnx")
args = parser.parse_args()

# logger
if os.path.exists(args.logdir):
    print("Removing old folder {}".format(args.logdir))
    shutil.rmtree(args.logdir)
if not os.path.exists(args.logdir):
    print("Creating new folder {}".format(args.logdir))
    os.makedirs(args.logdir)
print = utils.logger.info
 
# datasets
train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=args.ngpu)

# network
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(112)
net = model.alexnet().to(device)
# net = model.mlp(input_dims=784, n_hiddens=[256,256], n_classes=10)
# net = torch.nn.DataParallel(net, device_ids=range(args.ngpu))
# net.cuda()
optimizer = optim.SGD(net.parameters(), weight_decay=args.wd, lr=args.lr, momentum=0.9)
descreaing_lr = list(map(int, args.decreasing_lr.split(",")))

# train
best_acc, old_file = 0, None
t_begin = time.time()
try:
    # ready to do train
    for epoch in range(args.epochs):
        net.train()
        if epoch in descreaing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        for idx, (data, target) in enumerate(train_loader):
            acc_target = target.clone()
            if device == "cuda":
                data, target = data.cuda(), target.cuda()
            torch.set_grad_enabled(True)  

            optimizer.zero_grad()
            output = net(data)
            loss = F.cross_entropy(output, target)  # 交叉熵损失函数
            loss.backward()
            optimizer.step()

            if idx % args.log_interval == 0 and idx > 0:
                pred = output.data.max(1)[1]
                correct = pred.cpu().eq(acc_target).sum()
                acc = correct * 1.0 / len(data)  # epoch内统计
                print("Train Epoch {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr:{:.2e}".format(
                    epoch, idx * len(data), len(train_loader.dataset), loss.data, acc, 
                    optimizer.param_groups[0]['lr']
                ))
        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        print("Elapsed : {:.2f}s. {:.2f}s/epoch, {:.2f}s/batch, ets:{:.2f}s".format(
            elapse_time, speed_epoch, speed_epoch, eta))
        
        if epoch % args.test_interval == 0:
            net.eval()
            test_loss = 0
            correct = 0
            for data,target in test_loader:
                acc_target = target.clone()
                if device == "cuda":
                    data, target = data.cuda(), target.cuda()
                torch.set_grad_enabled(False)

                output = net(data)
                test_loss += F.cross_entropy(output, target)
                pred = output.data.max(1)[1]
                correct += pred.cpu().eq(acc_target).sum()
            test_loss = test_loss / len(test_loader)
            acc = 100. * correct / len(test_loader.dataset)
            print("\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss, correct, len(test_loader.dataset), acc))
            if acc > best_acc:
                new_file = os.path.join(args.logdir, "best-{}.pth".format(epoch))
                utils.model_snapshot(net, new_file, old_file=old_file, verbose=True)  # 保存权重
                best_acc = acc 
                old_file = new_file
except Exception as es:
    import traceback
    traceback.print_exc()
finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))

# onnx 转出
if args.onnx_root is not None:
    onnx_file = "model.onnx"
    dummpy_input = torch.randn(1, 1, 28, 28)
    utils.export2onnx(net.cpu(), old_file, os.path.join(args.onnx_root, onnx_file), dummpy_input)