import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from torchvision.transforms.transforms import RandomEqualize

class AttentionModule(nn.Module):

    # performs an up- and down- sampling by means of convolutional layers,
    # resulting in the same size as the original input.
    # up-sampling (transposed convolution) is applied first, then its output is activated
    # with a sigmoid. Afterwards, downsampling is performed in order to reshape tensors back to their original
    # shape. Finally, softmax is applied in order to provide a distribution over all the grid locations that
    # should serve as the Attention Map.

    def __init__(self, in_channels, n_steps=2):
        super(AttentionModule, self).__init__()
        self.upsamples = []
        self.downsamples = []
        upsample_inchannels = in_channels
        upsample_outchannels = in_channels//2
        for i in range(1, n_steps+1):
            self.upsamples.append((f"upconv{i}", nn.ConvTranspose2d(in_channels=upsample_inchannels, out_channels=upsample_outchannels, kernel_size=3)))
            self.upsamples.append((f"relu{i}", nn.ReLU(inplace=True)))
            upsample_inchannels = upsample_outchannels
            upsample_outchannels = upsample_inchannels//2
        
        downsample_inchannels = in_channels // (2**n_steps)
        downsample_outchannels = downsample_inchannels * 2
        for i in range(1, n_steps+1):
            self.downsamples.append((f"downconv{i}", nn.Conv2d(in_channels=downsample_inchannels, out_channels=downsample_outchannels, kernel_size=3)))
            self.downsamples.append((f"relu{i}", nn.ReLU(inplace=True)))
            downsample_inchannels *= 2
            downsample_outchannels *= 2
        
        self.upsamples = nn.Sequential(OrderedDict(self.upsamples))
        self.downsamples = nn.Sequential(OrderedDict(self.downsamples))

    def forward(self, x):
        x = self.upsamples(x)
        x = self.downsamples(x)
        B, _, H, W = x.shape
        x = torch.mean(x, dim=1)  # average along channels dimension
        x = torch.softmax(x.view(B, -1), dim=1)  # produce [0,1] attention map along the flattened dimension
        x = x.view(B, H, W)
        return x


class DeepAttentionClassifier(nn.Module):

    def __init__(self, pretrained):
        super(DeepAttentionClassifier, self).__init__()
        # initialize the baseline resnet for feature extraction
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-2]))  # get all children but the last two (replicate avg pooling below)
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # from here, also initialize binary attribute classifier (basic mlp)
        self.binary_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=13)
        )
        # we then need attention maps for the lower and upperbody portions
        self.upperbody_attn = AttentionModule(in_channels=512)
        self.lowerbody_attn = AttentionModule(in_channels=512)
        # and stacked mlp classifiers for both upperbody and lowerbody
        self.upperbody_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=80),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=80, out_features=9)
        )
        self.lowerbody_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=96),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=96, out_features=10)
        )

    def forward(self, x):
        # perform feature extraction with the backbone
        x = self.backbone(x)
        
        # pass the feature map to the binary mlp classifier
        x_bin = self.adaptive_avg_pool_2d(x) # same as last resnet pooling
        x_bin = x_bin.view(x_bin.shape[0], -1)  # keep the B dim and flatten the rest
        y_bin = self.binary_mlp(x_bin)

        # perform upperbody task with attention map
        up_attn = self.upperbody_attn(x)
        x_up = x * up_attn.unsqueeze(dim=1)  # hadamard product with the use of broadcasting
        x_up = self.adaptive_avg_pool_2d(x_up)  # dimensionality reduction
        x_up = x_up.view(x_up.shape[0], -1)  # flattening
        y_up = self.upperbody_mlp(x_up)

        # perform lowerbody task with attention map
        down_attn = self.lowerbody_attn(x)
        x_down = x * down_attn.unsqueeze(dim=1)
        x_down = self.adaptive_avg_pool_2d(x_down)
        x_down = x_down.view(x_down.shape[0], -1)
        y_down = self.lowerbody_mlp(x_down)

        # concatenate into [B, 32] output vector for compatibility with the previous training pipeline
        y = torch.cat(tensors=(y_bin, y_up, y_down), dim=1)
        return y


if __name__ == "__main__":
    img_tensor = torchvision.io.read_image("train_directory/0002_c3_027735928.jpg")/255
    img_tensor = img_tensor.unsqueeze(dim=0)
    deep_classifier = DeepAttentionClassifier(pretrained=True)
    deep_classifier(img_tensor)
    
