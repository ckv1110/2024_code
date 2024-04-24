import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.models.resnet as resnet

class DoubleConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, inplace=True):
        """
        A class representing a double convolutional block in a neural network model.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            padding (int, optional): The amount of padding to apply. Defaults to 1.
            bias (bool, optional): Whether to include bias in the convolutional layers. Defaults to False.
            inplace (bool, optional): Whether to perform the operations in-place. Defaults to True.
        """
        super(DoubleConvBlock, self).__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            BatchNormRelu(out_channels, inplace=inplace),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNormRelu(out_channels, inplace=inplace))
        
    def forward(self, inputs):
        """
        Forward pass of the double convolutional block.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        return self.conv_block(inputs)

class BatchNormRelu(torch.nn.Module):
    """
    A module that performs batch normalization followed by ReLU activation.

    Args:
        in_channels (int): The number of input channels.
        inplace (bool, optional): If set to True, ReLU activation is performed in-place. 
            Default is True.

    Returns:
        torch.Tensor: The output tensor after applying batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, inplace=True):
        super(BatchNormRelu, self).__init__()

        # Batch normalization and ReLU activation
        self.__bn_relu = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=inplace)
        )

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the model.

        """
        return self.__bn_relu(inputs)

class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, input_layer=False):
        """
        Initializes a ResidualBlock object.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): Stride of the convolution.
            padding (int or tuple): Padding added to the input during convolution.
            bias (bool): If True, adds a learnable bias to the output.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """
        super(ResidualBlock, self).__init__()

        if input_layer:
            self.__conv_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                BatchNormRelu(out_channels),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
            )

            # Shortcut connection
            self.__shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        else:
            self.__conv_block = torch.nn.Sequential(
                BatchNormRelu(in_channels),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                BatchNormRelu(out_channels),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
            )

            # Shortcut connection
            self.__shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

    def forward(self, inputs):
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the residual block.

        """
        return self.__conv_block(inputs) + self.__shortcut(inputs)

# Deprecated
class ResNetBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, downsample=None):
        """
        Initializes a ResNetBlock object.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): Stride of the convolution.
            padding (int or tuple): Padding added to the input.
            bias (bool): If True, adds a learnable bias to the output.
            downsample (torch.nn.Module, optional): Downsample layer. Defaults to None.
        """
        super(ResNetBlock, self).__init__()
        self.convblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            BatchNormRelu(out_channels, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.relu = torch.nn.ReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.convblock(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_size, st, pad, b, reverse_skip_concat=False):
        super(DecoderBlock, self).__init__()
        # print(f'in_channels = {in_channels}\nout_channels = {out_channels}')
        self.upscale = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=st)
        # self.upscale = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.residual_block = ResidualBlock(in_channels, out_channels, kernel_size=k_size, stride=st, padding=pad, bias=b)
        self.reverse_skip_concat = reverse_skip_concat

    def forward(self, inputs, skip):
        # print(f'inputs.shape = {inputs.shape}\nskip.shape = {skip.shape}')
        x = self.upscale(inputs)
        if x.shape != skip.shape: # Resize the encoder output to match the decoder input size if they don't match
                # encoder_output = TF.resize(encoder_output, size=x.shape[2:])
                # x = TF.resize(x, size=skip.shape[2:])
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                # encoder_output = F.interpolate(encoder_output, size=x.shape[2:], mode='bilinear', align_corners=False)
        # print(f'x.shape after upscale and resizing = {x.shape}')
        if self.reverse_skip_concat:
            x = torch.cat((x, skip), dim=1)
        else:
            x = torch.cat((skip, x), dim=1)
        x = self.residual_block(x)
        return x

class BuildUnet(torch.nn.Module):
    
    def __init__(self, in_channels, classes, reverse_skip_concat=False, k_size=3, st=2, pad=1, b=False):
        """
        Initializes the BuildUnet class.

        Args:
            in_channels (int): Number of input channels.
            classes (int): Number of output classes.
            reverse_skip_concat (bool, optional): Whether to concatenate the skip connections in reverse order. Defaults to False.
            k_size (int, optional): Kernel size for convolutional layers. Defaults to 3.
            s (int, optional): Stride for max pooling and transpose convolutional layers. Defaults to 2.
            pad (int, optional): Padding size for convolutional layers. Defaults to 1.
            b (bool, optional): Whether to use bias in convolutional layers. Defaults to False.
        """
        super(BuildUnet, self).__init__()
        self.layers_channels = [in_channels, 64, 128, 256, 512, 1024]
        self.reverse_skip_concat = reverse_skip_concat

        self.encoder_down_layers = torch.nn.ModuleList([DoubleConvBlock(c_in, c_out, k_size, pad, b)
                                                        for c_in, c_out in zip(self.layers_channels[:-1], self.layers_channels[1:])])
        self.maxpool_2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=st, padding=0)
        self.up_transpose_2x2 = torch.nn.ModuleList([torch.nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=st) 
                                                     for c_in, c_out in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])       
        self.decoder_up_layers = torch.nn.ModuleList([DoubleConvBlock(c_in, c_out, k_size, pad, b) 
                                                      for c_in, c_out in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
        
        self.final_conv = torch.nn.Conv2d(self.layers_channels[1], classes, kernel_size=1)

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
                x = F.interpolate(x, size=encoder_output.shape[2:], mode='bilinear', align_corners=False)
            
            if self.reverse_skip_concat:
                concat = torch.cat((x, encoder_output), dim=1)
            else:
                concat = torch.cat((encoder_output, x), dim=1)
            x = decoder_layer(concat)
        
        x = self.final_conv(x)

        return x

class BuildResidualUnet(torch.nn.Module):
    def __init__(self, in_channels, classes, k_size=3, st=2, pad=1, b=False, reverse_skip_concat=False):
        """
        Initializes the BuildResidualUnet model.

        Args:
            in_channels (int): Number of input channels.
            classes (int): Number of output classes.
            k_size (int, optional): Kernel size for convolutional layers. Defaults to 3.
            st (int, optional): Stride for convolutional layers. Defaults to 2.
            pad (int, optional): Padding for convolutional layers. Defaults to 1.
            b (bool, optional): Whether to include bias in convolutional layers. Defaults to False.
            reverse_skip_concat (bool, optional): Whether to concatenate skip connections in reverse order. Defaults to False.
        """
        super(BuildResidualUnet, self).__init__()

        self.layers_channels = [in_channels, 64, 128, 256, 512, 1024]
        self.encoder_layers = torch.nn.ModuleList([ResidualBlock(c_in, c_out, kernel_size=k_size, stride=st, padding=pad, bias=b) 
                                                   for c_in, c_out in zip(self.layers_channels[1:-1], self.layers_channels[2:])])
        self.input_layer = ResidualBlock(self.layers_channels[0], self.layers_channels[1], kernel_size=k_size, stride=st, padding=pad, bias=b, input_layer=True)
        self.encoder_layers.insert(0, self.input_layer)

        self.decoder_layers = torch.nn.ModuleList([DecoderBlock(c_in, c_out, k_size=k_size, st=1, pad=pad, b=b, reverse_skip_concat=reverse_skip_concat)
                                                    for c_in, c_out in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])

        self.final_conv = torch.nn.Conv2d(self.layers_channels[1], classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the BuildResUnet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        encoder_outputs = []
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            if i != len(self.encoder_layers) - 1:
                encoder_outputs.append(x)

        encoder_outputs = encoder_outputs[::-1] # Reverse the list to match the decoder layers for concatenation when creating skip connections

        for decoder_layer, encoder_output in zip(self.decoder_layers, encoder_outputs):
            x = decoder_layer(x, encoder_output)
        
        x = self.final_conv(x)
        
        return x

# class BuildResNetEncoderUnet(torch.nn.Module):