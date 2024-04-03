import torch
import torchvision.transforms.functional as TF

class BuildUnet(torch.nn.Module):
    """
    Basic U-Net model for multiclass semantic segmentation.
    
    Args:
        in_channels (int): Number of input channels.
        classes (int): Number of classes to predict.
        reverse_skip_concat (bool, optional): Whether to concatenate the encoder output with the decoder input in reverse order. Defaults to False.
    """
    
    def __init__(self, in_channels, classes, reverse_skip_concat=False):
        super(BuildUnet, self).__init__()
        self.layers_channels = [in_channels, 64, 128, 256, 512, 1024]
        # mid_channels = self.mid_channels
        self.reverse_skip_concat = reverse_skip_concat

        self.encoder_down_layers = torch.nn.ModuleList([self.__double_conv_layer(c_in, c_out) 
                                                        for c_in, c_out in zip(self.layers_channels[:-1], self.layers_channels[1:])])
        self.maxpool_2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.up_transpose_2x2 = torch.nn.ModuleList([torch.nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2) 
                                                     for c_in, c_out in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])       
        self.decoder_up_layers = torch.nn.ModuleList([self.__double_conv_layer(c_in, c_out) 
                                                      for c_in, c_out in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
        
        self.final_conv = torch.nn.Conv2d(self.layers_channels[1], classes, kernel_size=1)

    def __double_conv_layer(self, in_channels, out_channels):
        """
        Creates a double convolutional layer with batch normalization and ReLU activation.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of channels in the middle sub-layer. Defaults to None.
        
        Returns:
            torch.nn.Sequential: Double convolutional layer.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        encoder_outputs = []
        for i, encoder_layer in enumerate(self.encoder_down_layers):
            x = encoder_layer(x)
            if i != len(self.encoder_down_layers) - 1:
                encoder_outputs.append(x)
                x = self.maxpool_2x2(x)
        
        encoder_outputs = encoder_outputs[::-1] # Reverse the list to match the decoder layers for concatenation when creating skip connections

        for up_transpose, decoder_layer, encoder_output in zip(self.up_transpose_2x2, self.decoder_up_layers, encoder_outputs):
            x = up_transpose(x)

            if x.shape != encoder_output.shape: # Resize the encoder output to match the decoder input size if they don't match
                x = TF.resize(x, size=encoder_output.shape[2:])
            
            if self.reverse_skip_concat:
                concat = torch.cat((x, encoder_output), dim=1)
            else:
                concat = torch.cat((encoder_output, x), dim=1)
            x = decoder_layer(concat)
        
        x = self.final_conv(x)

        return x

class ResUnet(torch.nn.Module):
    def __init__(self, in_channels, classes):
        super(ResUnet, self).__init__()

        self.layers_channels = [in_channels, 64, 128, 256, 512, 1024]