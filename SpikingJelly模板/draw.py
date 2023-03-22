import torch
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import numpy as np

save_address = './result_pic/'

def draw_hot_pic(neurons, T):
    """
    输入为要绘制的神经元张量列表,格式为[T, N],值为电压
    根据神经元在不同时刻的电压绘制热力图
    """
    neurons = torch.cat(neurons).cpu()              # [100, 10] = [T, N]
    visualizing.plot_2d_heatmap(array=np.asarray(neurons), title='Membrane Potentials', xlabel='Simulating Step',
                                ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
    plt.savefig(save_address + 'Membrane_Potentials.svg', format='svg', dpi=300, bbox_inches='tight')


def draw_spike_frequency_histogram(firing_rate):
    """
    输入要绘制的神经元张量列表,格式为[T, N],值为脉冲发放频率
    绘制三维柱状图
    """
    visualizing.plot_2d_bar_in_3d(firing_rate.numpy(), title='spiking rates of output layer', xlabel='neuron index',
                                  ylabel='training epoch', zlabel='spiking rate', int_x_ticks=True, int_y_ticks=True,
                                  int_z_ticks=False, dpi=200)
    plt.savefig(save_address + 'spiking_rates_of_output_layer.svg', format='svg', dpi=300, bbox_inches='tight')


def draw_spike_v_histogram(v_list):
    """
    输入要绘制的神经元张量列表,格式为[T, N],值为电压
    绘制三维柱状图
    """
    visualizing.plot_2d_bar_in_3d(v_list, title='voltage of neurons', xlabel='neuron index', ylabel='simulating step',
                                  zlabel='voltage', int_x_ticks=True, int_y_ticks=True, int_z_ticks=False, dpi=200)
    plt.savefig(save_address + 'voltage_of_neurons.svg', format='svg', dpi=300, bbox_inches='tight')


def spike_out_circumstances(s_list):
    """
    输入要绘制的神经元张量列表,格式为[T, N],值为1或0
    绘制出N个神经元在T个时刻的脉冲发放时刻
    """
    visualizing.plot_1d_spikes(spikes = np.asarray(s_list), title='Membrane Potentials', xlabel='Simulating Step',
                               ylabel='Neuron Index', dpi=200)
    plt.savefig(save_address + 'Membrane_Potentials.svg', format='svg', dpi=300, bbox_inches='tight')


def show_feature_map(spike, nrows, ncols):
    """
    将C个尺寸为W*H的脉冲矩阵全部画出,然后排列成nrows行,ncols列.
    输入张量的尺寸为[C, W, H]
    """
    visualizing.plot_2d_feature_map(spike, nrows=nrows, ncols=ncols, space=2, title='Spiking Feature Maps', dpi=200)
    plt.savefig(save_address + 'Spiking_Feature_Maps.svg', format='svg', dpi=300, bbox_inches='tight')


def sgow_one_neuron_v_s(v_list, s_list, v_threshold, v_reset):
    """
    输入为格式为[T]的神经元不同时刻的电压和脉冲
    绘制单个神经元的电压和脉冲随时间的变化情况
    """
    visualizing.plot_one_neuron_v_s(v_list, s_list, v_threshold=v_threshold, v_reset=v_reset, dpi=200)
    plt.savefig(save_address + 'v_and_s_of_neuron.svg', format='svg', dpi=300, bbox_inches='tight')