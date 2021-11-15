from typing import Tuple, Union, List, Callable, Optional
import dataclasses
import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class TaskConfig:
    keyword: str = 'sheila'  # We will use 1 key word -- 'sheila'
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 10
    n_mels: int = 40
    cnn_out_channels: int = 8
    kernel_size: Tuple[int, int] = (5, 20)
    stride: Tuple[int, int] = (2, 8)
    hidden_size: int = 64
    gru_num_layers: int = 2
    bidirectional: bool = False
    num_classes: int = 2
    sample_rate: int = 16000
    max_window_length: int = 30
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

        
class Attention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)
        return (input * alpha).sum(dim=-2)

class CRNN(nn.Module):

    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
            config.stride[0] + 1
        
        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=0.1,
            bidirectional=config.bidirectional,
            batch_first = True
        )

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.stride = config.stride[1]
        self.streaming = False
        self.inp_buffer = None
        self.hidden = None
        self.max_window_length = config.max_window_length
    
    def forward(self, input):
        if self.streaming == False:
            input = input.unsqueeze(dim=1)
            conv_output = self.conv(input).transpose(-1, -2)
            gru_output, _ = self.gru(conv_output)
            contex_vector = self.attention(gru_output)
            output = self.classifier(contex_vector)
            return output
        else:
            input = input.unsqueeze(dim=1)
            if self.inp_buffer is None:
                self.inp_buffer = torch.Tensor([]).to(self.classifier.weight.device)
                self.gru_out_buffer = torch.Tensor([]).to(self.classifier.weight.device)
            batch = torch.cat((self.inp_buffer, input), -1)
            
            conv_output = self.conv(input).transpose(-1, -2)
            gru_output, self.hidden = self.gru(conv_output, self.hidden)
            self.inp_buffer = batch[:, :, :, self.stride * gru_output.size(0):]
            gru_output = torch.cat((self.gru_out_buffer, gru_output), 1)
            gru_output = gru_output[:, max(gru_output.shape[0] - self.max_window_length, 0):]
            self.gru_out_buffer = gru_output
            contex_vector = self.attention(gru_output)
            output = self.classifier(contex_vector)
            return output
        
    def clean_buffers(self):
        self.hidden = None
        self.inp_buffer = None
        self.gru_out_buffer = None
            
            
 
if __name__ == '__main__':

    config = TaskConfig()
    model = CRNN(config)
    model