
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        
        
        """
        conv1(out_channels=64, kernel_size=11, stride=4)+BN+ReLU
        -> maxpool1(kernel_size=3, stride=2)
        -> conv2(out_channels=128, kernel_size=5, padding=2)+BN+ReLU
        -> maxpool2(kernel_size=3, stride=2)
        -> conv3(out_channels=256, kernel_size=3, padding=1)+BN+ReLU
        -> conv4(out_channels=256, kernel_size=3, padding=1)+BN+ReLU
        -> conv5(out_channels=128, kernel_size=3, padding=1)+BN+ReLU
        -> maxpool2(kernel_size=3, stride=2)
        -> adaptive_avgpool(kernel_size=1x1)
        -> fc1(out_features=1024)+ReLU
        -> fc2(out_features=1024)+ReLU
        -> fc3(out_features=num_classes)
        """
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=outputs)

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomCNN2(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN2, self).__init__()
        
        
        """
        conv1(out_channels=64, kernel_size=11, stride=4)+BN+ReLU
        -> maxpool1(kernel_size=3, stride=2)
        -> conv2(out_channels=128, kernel_size=5, padding=2)+BN+ReLU
        -> maxpool2(kernel_size=3, stride=2)
        -> conv3(out_channels=256, kernel_size=3, padding=1)+BN+ReLU
        -> maxpool3(kernel_size=3, stride=2)
        -> conv4(out_channels=256, kernel_size=3, padding=1)+BN+ReLU
        -> conv5(out_channels=256, kernel_size=5, padding=2)+BN+ReLU
        -> maxpool4(kernel_size=3, stride=2)
        -> conv6(out_channels=256, kernel_size=3, padding=1)+BN+ReLU
        -> conv7(out_channels=128, kernel_size=3, padding=1)+BN+ReLU
        -> maxpool5(kernel_size=3, stride=2)
        -> adaptive_avgpool(kernel_size=1x1)
        -> fc1(out_features=1024)+ReLU+dropout
        -> fc2(out_features=1024)+ReLU+dropout
        -> fc3(out_features=num_classes)
        """
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=outputs)

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return x


class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']
        self.lstm_layers = 2

        if self.model_type == 'resnet50':
            self.encoder = resnet50(pretrained=True)
            
            # freeze the weights of the encoder
            for p in self.encoder.parameters():
                p.requires_grad = False
            
            self.encoder.fc = nn.Linear(2048, self.embedding_size)
        elif self.model_type == 'custom':
            self.encoder = CustomCNN(self.embedding_size)
        elif self.model_type == 'custom2':
            self.encoder = CustomCNN2(self.embedding_size)
            
        else:
            raise ValueError('Invalid model type')
        
        self.embedding = nn.Embedding(len(self.vocab), self.embedding_size)
        self.encoder_bn = nn.BatchNorm1d(self.embedding_size)
        self.decoder = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, num_layers=self.lstm_layers)
        self.fc = nn.Linear(self.hidden_size, len(self.vocab))


    def forward(self, images, captions, lengths, teacher_forcing=False):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        encoded_images = self.encoder_bn(self.encoder(images)) # (batch_size, embedding_size)
        encoded_captions = self.embedding(captions) # (batch_size, caption_length, embedding_size)
        
        batch_size = encoded_images.size(0)
        
        seq_length = encoded_captions.size(1) if teacher_forcing else self.max_length
        outputs = torch.zeros(batch_size, seq_length, len(self.vocab))
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size)
        
        if torch.cuda.is_available():
            outputs = outputs.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()
            
        h_c = (h0, c0)
        
        if teacher_forcing:
            
            # for i in range(seq_length):
            #     inputs = encoded_images if i == 0 else encoded_captions[:, i-1, :]
            #     output, h_c = self.decoder(inputs.unsqueeze(1), h_c)

            #     outputs[:, i, :] = self.fc(output.squeeze(1))
            
            # do the entire sequence all at once.
            inputs = torch.cat((encoded_images.unsqueeze(dim=1),encoded_captions), dim=1)
            packed = pack_padded_sequence(inputs, lengths, batch_first=True)
            lstm_out, h_c = self.decoder(packed)
            outputs = self.fc(lstm_out[0])
            
                                        
        else:
            #  deterministically decode the sequence

            inputs = encoded_images.unsqueeze(dim=1)
            
            generated_caption = []

            for i in range(self.max_length):
                lstm_out, h_c = self.decoder(inputs, h_c) # lstm_out is (batch_size, 1, hidden_size)
                output = self.fc(lstm_out.squeeze(dim=1)) # outputs is (batch_size, vocab_size)
                outputs[:, i, :] = output # store the output for this time step
                # token = torch.argmax(output, dim=1) # get the token with the highest probability (batch_size)
                
                if self.deterministic:
                    token = torch.argmax(output, dim=1)
                else: # randomly sample from the output distribution
                    token = torch.multinomial(F.softmax(output/self.temp, dim=1), num_samples=1).squeeze(dim=1)
                
                generated_caption.append(token) # store the token for this time step
                inputs = self.embedding(token).unsqueeze(dim=1) # inputs is (batch_size, 1, embedding_size)
                # print("input", inputs)
                
            # xy: outputs are also recorded, but not used.  Should they be used?
            
            res = torch.stack(generated_caption, dim=1) # res is (batch_size, max_length)
            return res

        return outputs


def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)
