import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import torchvision
# resnet = torchvision.models.resnet.resnet50(weights=None)

def create_layers(model_depth, starting_channel, array):
    """
    Create the layers for the U-Net model.

    Args:
        model_depth (int): Depth of the model.
        starting_channel (int): Starting number of channels.
        array (list): List to store the layers.

    Returns:
        list: List of layers.
    """
    for i in range(model_depth):
        if i == 0:
            array.append(starting_channel)
        else:
            array.append(array[-1] * 2)
    return array

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

class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, inplace=False):
        """
        Initializes a ConvBlock object.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            padding (int or tuple): Amount of padding to apply.
            bias (bool): Whether to include a bias term in the convolutional layer.
            inplace (bool, optional): Whether to perform the operation in-place. Defaults to True.
        """
        super(ConvBlock, self).__init__()
        self.__conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            BatchNormRelu(out_channels, inplace=inplace))
    
    def forward(self, inputs):
        return self.__conv_block(inputs)

class DoubleConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, inplace=False):
        """
        A class representing a double convolutional block in a neural network model.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel. Defaults to 3.
            padding (int): The amount of padding to apply. Defaults to 1.
            bias (bool): Whether to include bias in the convolutional layers.
            inplace (bool, optional): Whether to perform the operations in-place. Defaults to False.
        """
        super(DoubleConvBlock, self).__init__()
        self.__conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            BatchNormRelu(out_channels, inplace=inplace),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            BatchNormRelu(out_channels, inplace=inplace))
        
    def forward(self, inputs):
        """
        Forward pass of the double convolutional block.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        return self.__conv_block(inputs)

class DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_size, st, pad, b, reverse_skip_concat=False, residual=False):
        """
        Initializes a DecoderBlock object.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            k_size (int): Kernel size for the decoding block.
            st (int): Stride for the decoding block.
            pad (int): Padding for the decoding block.
            b (bool): Whether to include bias in the decoding block.
            reverse_skip_concat (bool, optional): Whether to concatenate the reverse skip connection. Defaults to False.
            residual (bool, optional): Whether to use residual connections. Defaults to False.
        """
        super(DecoderBlock, self).__init__()
        self.reverse_skip_concat = reverse_skip_concat
        self.__upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=st)
        if residual:
            self.__decoding_block = ResidualBlock(in_channels, out_channels, kernel_size=k_size, stride=st, padding=pad, bias=b)
        else:
            self.__decoding_block = DoubleConvBlock(in_channels, out_channels, kernel_size=k_size, padding=pad, bias=b, inplace=False)
        

    def forward(self, inputs, skip):
        # print(f'inputs.shape = {inputs.shape}\nskip.shape = {skip.shape}')
        x = self.__upsample(inputs)
        if x.shape != skip.shape: # Resize the encoder output to match the decoder input size if they don't match
                # encoder_output = TF.resize(encoder_output, size=x.shape[2:])
                # x = TF.resize(x, size=skip.shape[2:])
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                # encoder_output = F.interpolate(encoder_output, size=x.shape[2:], mode='bilinear', align_corners=False)
        if self.reverse_skip_concat:
            x = torch.cat((x, skip), dim=1)
        else:
            x = torch.cat((skip, x), dim=1)
        x = self.__decoding_block(x)
        return x

class ResNetDecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_size, pad, b, up_conv_in_channels=None, up_conv_out_channels=None, reverse_skip_concat=False, inplace=False):
        """
        Initializes the ResNetDecoderBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            k_size (int): Kernel size for the convolutional layers.
            st (int): Stride for the transpose convolutional layer.
            pad (int): Padding for the convolutional layers.
            b (bool): Whether to include bias in the convolutional layers.
            up_conv_in_channels (int, optional): Number of input channels for the transpose convolutional layer. Defaults to None.
            up_conv_out_channels (int, optional): Number of output channels for the transpose convolutional layer. Defaults to None.
            reverse_skip_concat (bool, optional): Whether to concatenate the skip connections in reverse order. Defaults to False.
        """
        super(ResNetDecoderBlock, self).__init__()
        self.reverse_skip_concat = reverse_skip_concat
        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        self.__upsample = torch.nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        self.__conv_block = DoubleConvBlock(in_channels, out_channels, kernel_size=k_size, padding=pad, bias=b, inplace=inplace)

    def forward(self, inputs, skip):
        x = self.__upsample(inputs)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if self.reverse_skip_concat:
            print(f"x shape: {x.shape}, skip shape: {skip.shape}")
            x = torch.cat((x, skip), dim=1)
        else:
            print(f"x shape: {x.shape}, skip shape: {skip.shape}")
            x = torch.cat((skip, x), dim=1)
                 
        x = self.__conv_block(x)
        return x
    
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

class BuildUnet(torch.nn.Module):
    
    def __init__(self, in_channels, classes, model_depth, input_layer_channels, reverse_skip_concat=False, k_size=3, st=2, pad=1, b=False):
        """
        Initializes the BuildUnet class.

        Args:
            in_channels (int): Number of input channels.
            classes (int): Number of output classes.
            model_depth (int): Depth of the model.
            input_layer_channels (int): Starting number of channels in the input layer.
            reverse_skip_concat (bool, optional): Whether to use reverse skip concatenation. Defaults to False.
            k_size (int, optional): Kernel size for convolutional layers. Defaults to 3.
            st (int, optional): Stride for max pooling and transpose convolutional layers. Defaults to 2.
            pad (int, optional): Padding for convolutional layers. Defaults to 1.
            b (bool, optional): Whether to use bias in convolutional layers. Defaults to False.
        """
        super(BuildUnet, self).__init__()
        self.layers_channels = [in_channels]
        for i in range(model_depth):
            if i == 0:
                self.layers_channels.append(input_layer_channels)
            else:
                self.layer = self.layers_channels[-1] * 2
                self.layers_channels.append(self.layer)
        # self.layers_channels = [in_channels, 64, 128, 256, 512, 1024]
        self.reverse_skip_concat = reverse_skip_concat

        self.encoder_down_layers = torch.nn.ModuleList([DoubleConvBlock(c_in, c_out, k_size, pad, b, inplace=True)
                                                        for c_in, c_out in zip(self.layers_channels[:-1], self.layers_channels[1:])])
        self.maxpool_2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=st, padding=0)
        self.up_transpose_2x2 = torch.nn.ModuleList([torch.nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=st) 
                                                     for c_in, c_out in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])       
        self.decoder_up_layers = torch.nn.ModuleList([DoubleConvBlock(c_in, c_out, k_size, pad, b, inplace=True) 
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
    def __init__(self, in_channels, classes, model_depth, input_layer_channels, k_size=3, st=1, pad=1, b=False, reverse_skip_concat=False):
        """
        Initialize the BuildResidualUnet class.

        Args:
            in_channels (int): Number of input channels.
            classes (int): Number of output classes.
            model_depth (int): Depth of the model.
            input_layer_channels (int): Starting number of channels in the input layer.
            k_size (int, optional): Kernel size for convolutional layers. Defaults to 3.
            st (int, optional): Stride for convolutional layers. Defaults to 2.
            pad (int, optional): Padding for convolutional layers. Defaults to 1.
            b (bool, optional): Bias term for convolutional layers. Defaults to False.
            reverse_skip_concat (bool, optional): Whether to concatenate skip connections in reverse order. Defaults to False.
        """
        super(BuildResidualUnet, self).__init__()
        self.layers_channels = create_layers(model_depth, input_layer_channels, [in_channels])
        
        # for i in range(model_depth):
        #     if i == 0:
        #         self.layers_channels.append(input_layer_channels)
        #     else:
        #         self.layer = self.layers_channels[-1] * 2
        #         self.layers_channels.append(self.layer)

        # self.layers_channels = [in_channels, 64, 128, 256, 512, 1024]
        self.encoder_layers = torch.nn.ModuleList([ResidualBlock(c_in, c_out, kernel_size=k_size, stride=st, padding=pad, bias=b) 
                                                   for c_in, c_out in zip(self.layers_channels[1:-1], self.layers_channels[2:])])
        self.input_layer = ResidualBlock(self.layers_channels[0], self.layers_channels[1], kernel_size=k_size, stride=st, padding=pad, bias=b, input_layer=True)
        self.encoder_layers.insert(0, self.input_layer)

        self.decoder_layers = torch.nn.ModuleList([DecoderBlock(c_in, c_out, k_size=k_size, st=1, pad=pad, b=b, reverse_skip_concat=reverse_skip_concat, residual=True)
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

class BuildResNetEncoderUnet(torch.nn.Module):
    
    def __init__(self, in_channels, classes, resnet_type, weights, k_size=3, pad=1, b=False, reverse_skip_concat=False, residual_decoder=False):
        super(BuildResNetEncoderUnet, self).__init__()
        if resnet_type == 'resnet18':
            self.resnet_encoder = resnet.resnet18(weights=weights)
            self.depth = 6
            self.layers_channels = [3, 64, 64, 128, 256, 512]
        elif resnet_type == 'resnet34':
            self.resnet_encoder = resnet.resnet34(weights=weights)
            self.depth = 5
            self.layers_channels = [3, 64, 64, 128, 256, 512]
        elif resnet_type == 'resnet50':
            self.resnet_encoder = resnet.resnet50(weights=weights)
            self.depth = 6
            self.starting_channels = 64
            self.layers_channels = create_layers(self.depth, self.starting_channels, [in_channels])
        elif resnet_type == 'resnet101':
            self.resnet_encoder = resnet.resnet101(weights=weights)
            self.depth = 6
            self.starting_channels = 64
            self.layers_channels = create_layers(self.depth, self.starting_channels, [in_channels])
        elif resnet_type == 'resnet152':
            self.resnet_encoder = resnet.resnet152(weights=weights)
            self.depth = 6
            self.starting_channels = 64
            self.layers_channels = create_layers(self.depth, self.starting_channels, [in_channels])

        self.input_layer = torch.nn.Sequential(*list(self.resnet_encoder.children())[:3])
        self.input_maxpool = list(self.resnet_encoder.children())[3]
        
        self.encoder_layers = torch.nn.ModuleList([l for l in list(self.resnet_encoder.children()) if isinstance(l, torch.nn.Sequential)])

        self.bridge_layer = DoubleConvBlock(self.layers_channels[-1], self.layers_channels[-1], kernel_size=k_size, padding=pad, bias=b, inplace=False)
        
        self.decoder_layers = torch.nn.ModuleList([ResNetDecoderBlock(c_in, c_out, k_size=k_size, pad=pad, b=b, reverse_skip_concat=reverse_skip_concat)
                                            for c_in, c_out in zip(self.layers_channels[::-1][:3], self.layers_channels[::-1][1:4])])

        self.decoder_layers.append(ResNetDecoderBlock(in_channels = self.layers_channels[2] + self.layers_channels[1], out_channels = self.layers_channels[2],
                                                          up_conv_in_channels = self.layers_channels[3], up_conv_out_channels = self.layers_channels[2],
                                                            k_size=k_size, pad=pad, b=b, reverse_skip_concat=reverse_skip_concat))
        self.decoder_layers.append(ResNetDecoderBlock(in_channels = self.layers_channels[1]+self.layers_channels[0], out_channels = self.layers_channels[1],
                                                          up_conv_in_channels = self.layers_channels[2], up_conv_out_channels = self.layers_channels[1],
                                                            k_size=k_size, pad=pad, b=b, reverse_skip_concat=reverse_skip_concat))

        self.final_conv = torch.nn.Conv2d(self.layers_channels[1], classes, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        encoder_outputs.append(x)
        x = self.input_layer(x)
        encoder_outputs.append(x)
        x = self.input_maxpool(x)

        for i, encoder_layer in enumerate(self.encoder_layers):
                x = encoder_layer(x)
                if i != len(self.encoder_layers) - 1:
                    encoder_outputs.append(x)
        
        x = self.bridge_layer(x)
        encoder_outputs = encoder_outputs[::-1]

        for decoder_layer, encoder_output in zip(self.decoder_layers, encoder_outputs):
            x = decoder_layer(x, encoder_output)
        
        x = self.final_conv(x)

        return x
