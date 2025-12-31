import torch
import torch.nn as nn
from config import STAGES_CONFIG

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class SearchableMBConv(nn.Module):

    def __init__(self, inp, oup, stride):
        super(SearchableMBConv, self).__init__()
        self.stride = stride
        self.expansion_choices = [3, 6]
        self.kernel_choices = [3, 5]
        
        self.ops = nn.ModuleList()
        for exp_ratio in self.expansion_choices:
            for kernel_size in self.kernel_choices:
                hidden_dim = inp * exp_ratio
                padding = (kernel_size - 1) // 2
                op = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                )
                self.ops.append(op)
        
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x, choice_idx):
        op_to_use = self.ops[choice_idx]
        if self.use_res_connect:
            return x + op_to_use(x)
        else:
            return op_to_use(x)

class SuperNet(nn.Module):
    
    def __init__(self, num_classes):
        super(SuperNet, self).__init__()
        self.num_classes = num_classes
        self.input_conv = conv_bn(3, 32, 2) #input, no change, fixed rahegi

        self.stages = nn.ModuleList() # state config wali list hai
        inp = 32
        for oup, depth_choices, stride in STAGES_CONFIG:
            stage_blocks = nn.ModuleList()
            max_depth = max(depth_choices) # jo bhi max depth select kar sakta hai chahe nas koi use kare
            for i in range(max_depth):
                current_stride = stride if i == 0 else 1
                stage_blocks.append(SearchableMBConv(inp, oup, current_stride))
                inp = oup
            self.stages.append(stage_blocks) # ab supernet ke pass 5 stages
        # upar loop se hum bohot sare block banayenge jinke inside unke bhi versions honge
        self.output_conv = conv_bn(STAGES_CONFIG[-1][0], 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, self.num_classes)
        # upar wali mobilenet ki same hai paper se uthai hai
    def forward(self, x, architecture):
        
        x = self.input_conv(x)
        for stage_idx, stage_blocks in enumerate(self.stages):
            stage_architecture = architecture[stage_idx]
            num_blocks_in_stage = len(stage_architecture)
            for block_idx in range(num_blocks_in_stage):
                op_choice = stage_architecture[block_idx]
                block = stage_blocks[block_idx]
                x = block(x, op_choice)
        x = self.output_conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

