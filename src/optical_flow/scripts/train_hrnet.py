import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from optical_flow.models.hrnet.cls_hrnet import HighResolutionNet
from optical_flow.models.datasets import YoutubeDataset
from optical_flow.settings import hrnet_w18_cfg

class HighResolutionNetEncoder(HighResolutionNet):

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Keep Decoder Part of Classification Head which will be retrained
        return y_list


def test_image_sizes():
    dataset = YoutubeDataset()
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0)
    print(next(iter(dataloader)).shape)


def test_load_hr_net():
    hrnet = HighResolutionNetEncoder(hrnet_w18_cfg)
    hrnet.load_state_dict(torch.load('./models/hrnet/pretrained/hrnet_w18_small_model_v2.pth', map_location=torch.device('cpu')))
    # drop last layer of hrnet-classification
    for param in hrnet.parameters():
        param.requires_grad = False

    dataset = YoutubeDataset("frames480")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images = next(iter(dataloader))
    images = images.float()

    enc_out = hrnet.forward(images)

    decoder = smp.unet.model.UnetDecoder(encoder_channels=[18, 36, 72, 144], decoder_channels=[2, 3, 4, 5], n_blocks=4)
    decoder.forward(enc_out)


def main():
    #test_image_sizes()
    test_load_hr_net()


if __name__ == '__main__':
    main()