import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


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
            self.upsamples.append((f"bn{i}", nn.BatchNorm2d(num_features=upsample_outchannels)))
            self.upsamples.append((f"tanh{i}", nn.Tanh()))
            upsample_inchannels = upsample_outchannels
            upsample_outchannels = upsample_inchannels//2
        
        downsample_inchannels = in_channels // (2**n_steps)
        downsample_outchannels = downsample_inchannels * 2
        for i in range(1, n_steps+1):
            self.downsamples.append((f"downconv{i}", nn.Conv2d(in_channels=downsample_inchannels, out_channels=downsample_outchannels, kernel_size=3)))
            self.downsamples.append((f"bn{i}", nn.BatchNorm2d(num_features=downsample_outchannels)))
            self.downsamples.append((f"tanh{i}", nn.Tanh()))
            downsample_inchannels *= 2
            downsample_outchannels *= 2
        
        self.upsamples = nn.Sequential(OrderedDict(self.upsamples))
        self.downsamples = nn.Sequential(OrderedDict(self.downsamples))

    def forward(self, x):
        x = self.upsamples(x)
        x = self.downsamples(x)
        B, C, H, W = x.shape
        x = torch.softmax(x.view(B, -1), dim=1)  # produce [0,1] attention map along the flattened dimension
        x = x.view(B, C, H, W)
        return x


class DeepAttentionClassifier(nn.Module):

    def __init__(self, pretrained):
        super(DeepAttentionClassifier, self).__init__()
        # initialize the baseline resnet for feature extraction
        self.backbone = torchvision.models.resnet34(pretrained=pretrained)
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-2]))  # get all children but the last two (replicate avg pooling below)
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # mlp for age attributes
        self.age_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=256, out_features=4),
            nn.Sigmoid()
        )
        # from here, also initialize binary attribute classifier (basic mlp)
        self.binary_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=9),
            nn.Sigmoid()
        )
        # we then need attention maps for the first attention branches
        self.upperbody_attn1 = AttentionModule(in_channels=512)
        self.lowerbody_attn1 = AttentionModule(in_channels=512)
        # 2nd branch attention maps
        self.upperbody_attn2 = AttentionModule(in_channels=512)
        self.lowerbody_attn2 = AttentionModule(in_channels=512)
        
        # and stacked mlp classifiers for both upperbody and lowerbody
        self.upperbody_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=256, out_features=9),
            nn.Sigmoid()
        )
        self.lowerbody_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=256, out_features=10),
            nn.Sigmoid()
        )

    def forward(self, x):
        # perform feature extraction with the backbone
        x = self.backbone(x)
        x_mlp = self.adaptive_avg_pool_2d(x) # same as last resnet pooling

        # feedforward (age mlp)
        x_age = x_mlp.view(x_mlp.shape[0], -1)  # keep the B dim and flatten the rest
        y_age = self.age_mlp(x_age)

        # pass the feature map to the binary mlp classifier
        x_bin = x_mlp.view(x_mlp.shape[0], -1)  # keep the B dim and flatten the rest
        y_bin = self.binary_mlp(x_bin)

        # first attention branch on upperbody
        up_attn1 = self.upperbody_attn1(x)
        x_up = x * (1 + up_attn1)  # hadamard product with the use of broadcasting
        # second attention branch on upperbody
        up_attn2 = self.upperbody_attn2(x_up)
        x_up = x_up * (1 + up_attn2)
        # multi layer perceptron for upperbody attribute classification
        x_up = self.adaptive_avg_pool_2d(x_up)  # dimensionality reduction
        x_up = x_up.view(x_up.shape[0], -1)  # flattening
        y_up = self.upperbody_mlp(x_up)

        # first attention branch on lowerbody clothing
        down_attn1 = self.lowerbody_attn1(x)
        x_down = x * (1 + down_attn1)
        # second attention branch on lowerbody clothing
        down_attn2 = self.lowerbody_attn2(x_down)
        x_down = x_down * (1 + down_attn2)
        # multi-layer perceptron for lowerbody attribute classification
        x_down = self.adaptive_avg_pool_2d(x_down)
        x_down = x_down.view(x_down.shape[0], -1)
        y_down = self.lowerbody_mlp(x_down)

        # concatenate into [B, 32] output vector for compatibility with the previous training pipeline
        y = torch.cat(tensors=(y_age, y_bin, y_up, y_down), dim=1)
        return y

    def inference(self, x):
        preds = self(x)
        age_preds = preds[:, :4]
        age_preds = torch.argmax(age_preds, dim=1, keepdim=True)
        independent_preds = preds[:, 4:13].round()
        up_preds = preds[:, 13:22].round()
        # up_preds = torch.argmax(up_preds, dim=1, keepdim=True)
        down_preds = preds[:, 22:].round()
        # down_preds = torch.argmax(down_preds, dim=1, keepdim=True)
        return torch.cat(tensors=(age_preds, independent_preds, up_preds, down_preds), dim=1) + 1


if __name__ == "__main__":
    img_tensor = torchvision.io.read_image("train_directory/0002_c3_027735928.jpg")/255
    img_tensor = img_tensor.unsqueeze(dim=0)
    deep_classifier = DeepAttentionClassifier(pretrained=True)
    deep_classifier(img_tensor)
    
