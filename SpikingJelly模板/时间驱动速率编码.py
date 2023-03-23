import torch
import torch.nn as nn
import torchvision
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import encoding
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda import amp
import os
import draw

parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

parser.add_argument('--device', default='cuda:0', help='运行的设备\n')
parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，\n')
parser.add_argument('--log-dir', default='./runs', help='保存tensorboard日志文件的位置.')
parser.add_argument('--resume-dir', type=str, default='./resume', help='输出断点续训等文件的位置\n')
parser.add_argument('--model-output-dir', default='./result_model', help='模型保存路径，例如“./”\n')
parser.add_argument('--num-workers', default=4, type=int, help='加载数据集使用的核心数量\n')
parser.add_argument('--lr-scheduler', default='CosALR', type=str, help='选择学习率衰减算法 StepLR or CosALR')
parser.add_argument('-T_max', default=32, type=int, help='使用余弦退火算法优化学习率时的最大优化次数(优化前多少个epochs)')
parser.add_argument('--seed', default=2023, type=int, help='使用的随机种子\n')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n')
parser.add_argument('-T', '--timesteps', default=100, type=int, dest='T', help='仿真时长，例如“100”\n')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n', dest='lr')
parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，')
parser.add_argument('-N', '--epoch', default=10, type=int, help='训练epoch\n')

parser.add_argument('--amp', action='store_true', help='是否启动混合精度训练')
parser.add_argument('--cupy', action='store_true', help='是否使用cupy和多步传播')
parser.add_argument('--lr-scheduler-flag', action='store_true', help='是否使用学习率自衰减')
parser.add_argument('--resume', action='store_true', help='是否使用断点续训')

def main():
    ''' 每个epoch中进行一次测试 '''

    args = parser.parse_args()
    print("############## 参数详情 ##############")
    # print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))     # 输出所有参数
    print("####################################")

    device = args.device
    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size
    lr = args.lr
    num_steps = args.T
    tau = args.tau
    epochs = args.epoch
    _seed_ = args.seed
    resume_dir = args.resume_dir

    writer = SummaryWriter(log_dir)         # 用于tensorboard
    train_batch_num = 0                     # 记录训练了多少个batch
    max_test_accuracy = 0
    train_batch_accs = []                   # 记录每个batch的训练准确率
    test_accs = []                          # 记录测试准确率
    draw_spike_frequency_histogram_data = []        # 用来绘制神经元的脉冲发放频率柱状图

    # 用于混合精度训练
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # 固定随机种子
    torch.manual_seed(_seed_)  # 为CPU和CUDA设置种子用于生成随机数，使结果固定
    np.random.seed(_seed_)
    torch.backends.cudnn.deterministic = True  # 使cuda使用同样的的核心分配方法
    torch.backends.cudnn.benchmark = False  # 不为卷积等运算进行硬件层面上的优化

    train_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transforms.ToTensor())

    # 取少部分数据集用来调试程序
    train_dataset, _ = data.random_split(train_dataset, [100, len(train_dataset)-100])
    test_dataset, _ = data.random_split(test_dataset, [40, len(test_dataset)-40])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=args.num_workers)

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10, bias=False),
        neuron.LIFNode(tau=tau)
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    encoder = encoding.PoissonEncoder()         # 使用泊松编码器

    # 学习率自衰减
    if args.lr_scheduler_flag:              # 如果启用学习率自衰减
        lr_scheduler = None
        if args.lr_scheduler == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'CosALR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
        else:
            raise NotImplementedError(args.lr_scheduler)

    # 断点续训
    if args.resume:
        checkpoint = torch.load(os.path.join(resume_dir, 'checkpoint_latest.pth'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.lr_scheduler_flag:              # 如果设置了权重自衰减
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        train_correct_sum_in_epoch = 0        # 每个epoch用于训练的数据中预测正确的数量
        train_data_sum_in_epoch = 0           # 每个epoch用于训练的数据总量
        print(f"train epoch : {epoch}")
        net.train()
        for img, label in tqdm(train_loader):                       # 循环次数 ≈ 数据总量/batch_size
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()            # 对标签进行one-hot编码以便后面计算损失
            # 根据是否开启混合精度训练，选择不同的训练方法
            if args.amp:        # 混合精度训练
                with amp.autocast():
                    for step in range(num_steps):
                        if step == 0:
                            # print(encoder(img).shape)                     # [64, 1, 28, 28]
                            train_data_out_spike_counter_in_num_step = net(encoder(img).float())  # 记录整个num_steps内输出层神经元的spike次数
                            # print(result.shape)                           # result的格式为[batch_size, 类别数量]
                        else:
                            spike_num = net(encoder(img).float())  # 在w X h维度上相加
                            train_data_out_spike_counter_in_num_step += spike_num
                    out_spike_counter_frequency = train_data_out_spike_counter_in_num_step / num_steps  # 在num_steps内的脉冲发放频率
                    loss = F.mse_loss(out_spike_counter_frequency, label_one_hot)  # 输出层神经元的脉冲发放频率与真实类别的MSE
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:       # 正常训练
                for step in range(num_steps):
                    if step == 0:
                        # print(encoder(img).shape)                     # [64, 1, 28, 28]
                        train_data_out_spike_counter_in_num_step = net(encoder(img).float())              # 记录整个num_steps内输出层神经元的spike次数
                        # print(result.shape)                           # result的格式为[batch_size, 类别数量]
                    else:
                        spike_num = net(encoder(img).float())  # 在w X h维度上相加
                        train_data_out_spike_counter_in_num_step += spike_num
                out_spike_counter_frequency = train_data_out_spike_counter_in_num_step / num_steps             # 在num_steps内的脉冲发放频率
                loss = F.mse_loss(out_spike_counter_frequency, label_one_hot)           # 输出层神经元的脉冲发放频率与真实类别的MSE
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            functional.reset_net(net)                                   # 重置网络状态

            # 累加获得整个epoch的准确率
            train_correct_sum_in_epoch += (out_spike_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()       # 格式为[batch_size]，值为0或1
            # print(out_spike_counter_frequency.max(1)[1].shape)          # out_spike_counter_frequency.max(1)[1]的值为每个图片的类别组成的列表，形状为[batch_size]
            train_data_sum_in_epoch += label.numel()                                  # numel函数返回数组中的元素个数,值为总图片数量

            # 单次记录每个batch的准确率
            train_batch_accuracy = (out_spike_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()        # 每个batch中的accurary
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_batch_num)        # 存储
            train_batch_accs.append(train_batch_accuracy)                                       # 存储
            train_batch_num += 1            # 几个batch
        train_accuracy_in_epoch = train_correct_sum_in_epoch / train_data_sum_in_epoch           # 每个epoch的accurary
        if args.lr_scheduler_flag:
            lr_scheduler.step()             # 根据选择的优化算法，对学习率进行调整，每个epoch执行一次

        print("###############   End of training, start testing   ###############")
        net.eval()
        with torch.no_grad():              # 每个epoch进行一次测试
            test_data_correct_sum = 0      # 测试过程中模型输出正确的数量
            test_data_sum = 0              # 测试过程中的总数据量
            for img, label in tqdm(test_loader):
                img = img.to(device)
                label = label.to(device)
                for step in range(num_steps):
                    if step == 0:
                        test_data_out_spike_counter = net(encoder(img).float())             # 记录输出层的spike
                    else:
                        test_data_out_spike_counter = net(encoder(img).float())

                test_data_correct_sum += (test_data_out_spike_counter.max(1)[1] == label.to(device)).float().sum().item()       # 累加正确数量
                test_data_sum += label.numel()      # 累加总数据量
                functional.reset_net(net)
            test_accuracy = test_data_correct_sum / test_data_sum       # 测试准确率
            writer.add_scalar('test_accurary', test_accuracy, epoch)    # 存储测试accurary
            test_accs.append(test_accuracy)                             # 存储测试accurary
            # 判断本次结果是不是目前最好的结果
            save_max = False
            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy       # 存储最高的准确率
                save_max = True                         # 在后面输出为当前最好模型
        print(f"epoch{epoch}中，训练集准确率为{train_accuracy_in_epoch}, 测试准确率为{test_accuracy},至今最好的测试准确率为{max_test_accuracy},共训练了{train_batch_num}个batch")

        # 保存断点续训所需的记录训练进度的文件
        if args.lr_scheduler_flag:              # 当启用学习率衰减时
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_accuracy}
        else:                                   # 没有启用学习率衰减时
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_accuracy}

        if save_max:
            torch.save(checkpoint, os.path.join(resume_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(resume_dir, 'checkpoint_latest.pth'))

    # 保存模型
    # torch.save(net, model_output_dir + 'SpikingJellyDemo.ckpt')
    # 读取模型
    # net = torch.load(model_output_dir + 'SpikingJellyDemo.ckpt')

    # 下面为保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net[-1]      # 输出层
    output_layer.v_seq = []     # 最后一层的电压
    output_layer.s_seq = []     # 最后一层的脉冲
    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))            # 存储最后一层的电压
        m.s_seq.append(y.unsqueeze(0))              # 存储最后一层的脉冲（0/1）
    output_layer.register_forward_hook(save_hook)   # 注册钩子

    with torch.no_grad():                       # 使用一张图片输入网络来测试结果
        img, label = test_dataset[0]
        img = img.to(device)
        draw_hot_list = []                      # 用来存储神经元在不同时刻的电压以用来绘制热力图
        for step in range(num_steps):           # 使用测试集中的一张图片测试放电率，由于钩子的存在，每个时间步都会记录电压和脉冲
            if step == 0:
                out_spike_counter = net(encoder(img).float())       # 格式为[1, 1]
            else:
                out_spike_counter += net(encoder(img).float())
            draw_hot_list.append(net[2].v)             # 记录每一个神经元(各个通道中)在每个时刻的电压
        draw.draw_hot_pic(draw_hot_list, num_steps)                 # 绘制并保存热力图
        out_spike_counter_frequency = (out_spike_counter / num_steps).cpu().numpy()
        print(f'Firing rate: {out_spike_counter_frequency}')                    # 输出每个神经元的放电率（10个）

        output_layer.v_seq = torch.cat(output_layer.v_seq)                      # cat前为长度为100的列表，cat后为张量[100, 1, 10]
        output_layer.s_seq = torch.cat(output_layer.s_seq)

        v_t_array = output_layer.v_seq.cpu().numpy().squeeze().T        # .T代表转置，v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy", v_t_array)         # 存储电压值
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze().T        # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy", s_t_array)         # 存储输出

    train_accs = np.array(train_batch_accs)         # 每个batch的训练准确率
    np.save('train_accs.npy', train_accs)           # 以二进制文件格式存储每个batch的训练准确率列表
    test_accs = np.array(test_accs)
    np.save('test_accs.npy', test_accs)             # 以二进制文件格式存储每个epoch对测试集的准确率列表


if __name__ == '__main__':
    main()