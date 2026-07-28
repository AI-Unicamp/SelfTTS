import torch
import torch.nn as nn

from torch.autograd import Function


class LinearNorm(nn.Module):
    ''' Linear Norm Module:
        - Linear Layer
    '''
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        ''' Forward function of Linear Norm
            x = (*, in_dim)
        '''
        x = self.linear_layer(x)  # (*, out_dim)
        
        return x

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    ''' Gradient Reversal Layer
            Y. Ganin, V. Lempitsky,
            "Unsupervised Domain Adaptation by Backpropagation",
            in ICML, 2015.
        Forward pass is the identity function
        In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    '''

    def __init__(self, lambda_reversal=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ConvEncoder(nn.Module):

    def __init__(self, embed_dim, out_dim):
        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        )

    def forward(self, x):
        ''' Forward function of ConvEncoder:
            x = (B, embed_dim, len)
        '''
        # pass through encoder
        outputs = self.encoder(x)  # (B, out_dim, len)
        outputs = torch.mean(outputs, dim=-1) # (B, out_dim)
        return outputs

class ConvEncoderClassifier(nn.Module):

    def __init__(self, embed_dim, out_dim):
        super(ConvEncoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        )

        self.classifier = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        ''' Forward function of ConvEncoderClassifier:
            x = (B, embed_dim, len)
        '''
        # pass through encoder
        outputs = self.encoder(x)  # (B, out_dim, len)
        outputs = torch.mean(outputs, dim=-1) # (B, embed_dim)

        outputs = self.classifier(outputs)  # (B, out_dim)

        return outputs

class LinearEncoder(nn.Module):
    ''' LinearEncoder Module:
        - 3x Linear Layers with ReLU
    '''
    def __init__(self, embed_dim, out_dim):
        super(LinearEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, out_dim, w_init_gain='linear')
        )
    
    def forward(self, x):
        ''' Forward function of LinearEncoder:
            x = (B, out_dim)
        '''
        # pass through encoder
        outputs = self.encoder(x)  # (B, out_dim)
        
        return outputs

class LinearEncoderClassifier(nn.Module):
    ''' LinearEncoderClassifier Module:
        - 3x Linear Layers with ReLU
    '''
    def __init__(self, embed_dim, out_dim):
        super(LinearEncoderClassifier, self).__init__()
        
        self.encoder = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, out_dim, w_init_gain='linear')
        )
    
    def forward(self, x):
        ''' Forward function of LinearEncoderClassifier:
            x = (B, out_dim)
        '''
        # pass through encoder
        outputs = self.encoder(x)  # (B, out_dim)
        
        return outputs

class LinearEncoderClassifierE3(nn.Module):
    ''' LinearEncoderClassifier Module:
        - 3x Linear Layers with ReLU
    '''
    def __init__(self, embed_dim, out_dim):
        super(LinearEncoderClassifierE3, self).__init__()
        
        self.encoder = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            LinearNorm(embed_dim, out_dim, w_init_gain='linear'),
        )
    
    def forward(self, x):
        ''' Forward function of LinearEncoderClassifier:
            x = (B, out_dim)
        '''
        # pass through encoder
        outputs = self.encoder(x)  # (B, out_dim)
        
        return outputs

class LinearEncoderClassifierE3Pooled(nn.Module):
    ''' LinearEncoderClassifier Module:
        - 3x Linear Layers with ReLU
    '''
    def __init__(self, embed_dim, out_dim):
        super(LinearEncoderClassifierE3Pooled, self).__init__()
        
        self.encoder = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            LinearNorm(embed_dim, out_dim, w_init_gain='linear'),
        )
    
    def forward(self, x):
        ''' Forward function of LinearEncoderClassifier:
            x = (B, L, out_dim)
        '''
        x = torch.mean(x, dim=-1) # (B, embed_dim)
        # pass through encoder
        outputs = self.encoder(x)  # (B, out_dim)
        
        return outputs