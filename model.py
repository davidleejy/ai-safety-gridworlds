import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym
import numpy as np
from relational_module import MultiHeadAttention

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def xy_meshgrid(height, width):
    step = 2.0/(width-1)
    eps = step / 10.0
    xs = np.arange(-1.0, 1.0+eps, step)
    step = 2.0/(height-1)
    eps = step / 10.0
    ys = np.arange(-1.0, 1.0+eps, step)
    xv, yv = np.meshgrid(xs, ys)    
    return torch.Tensor(xv), torch.Tensor(yv)

class ACModel_Relational(nn.Module, torch_rl.RecurrentACModel):
    '''
    Actor-critic model with a relational module.
    '''
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        # self.use_text = use_text
        self.use_memory = use_memory
        self.recurrent = use_memory

        # Define image embedding
        image_chans = obs_space["image"][2]
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_chans, 8, (1, 1)),
            nn.ReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, (1, 1)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64   # original. image_embedding_size is basically the number of elements at output of self.image_conv(x), not accounting for batch size.
        self.image_embedding_size = n * m * 16

        # Define x, y position layers
        x_layer, y_layer = xy_meshgrid(height=n, width=m)
        self.x_layer = x_layer.view(1, 1, *(x_layer.shape)) # shape (1,1,h,w)
        self.y_layer = y_layer.view(1, 1, *(y_layer.shape))
        if torch.cuda.is_available():
            self.x_layer = self.x_layer.cuda()
            self.y_layer = self.y_layer.cuda()
        # print(self.x_layer.shape)
        # print(self.x_layer)
        # print(self.y_layer)
        # print("-"*9)

        # Define relational modules
        self.relational_block = MultiHeadAttention(n_heads=6, dk=5, dv=6, lq=16+2, lk=16+2, lv=16+2)

        # Feature-wise max pool
        self.feature_wise_maxpool = nn.MaxPool1d(kernel_size=n*m) # TODO could this be position-wise maxpool?

        # MLP after feature-wise/position-wise maxpool
        self.mlp_aft_feat_wise_maxpool = nn.Sequential(
            nn.Linear(16+2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        ) # 2 hidden layers
        self.actorcritic_input_size = 32

        # Define memory
        if self.use_memory:
            NotImplementedError
            # self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        # self.embedding_size = self.semi_memory_size
        
        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.actorcritic_input_size, 16),
                nn.Tanh(),
                nn.Linear(16, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.actorcritic_input_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None):
        
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        # print('fx shape input', x.shape)
        x = self.image_conv(x)
        # x = x.reshape(x.shape[0], -1)
        
        # x = x.squeeze()
        assert len(x.shape) == 4 # current implementation of relational module accepts nchw format

        # print(x.shape) # shape bs x chans x h x w
        
        x_layer = self.x_layer.clone()
        y_layer = self.y_layer.clone()
        x_layer = x_layer.repeat(x.shape[0],1,1,1)
        y_layer = y_layer.repeat(x.shape[0],1,1,1)
        x_tagged = torch.cat([x, x_layer, y_layer], dim=-3)
        x_tagged = x_tagged.view(x_tagged.shape[0], x_tagged.shape[1], x_tagged.shape[2]*x_tagged.shape[3]) # shape bs x chans x h*w
        x_tagged = x_tagged.permute(0,2,1) # shape bs x (h*w) x chans
        # print(x_tagged.shape)
        #watn x_tagged to be bs x h*w x chans

        x_attn, attention = self.relational_block(x_tagged, x_tagged, x_tagged) # x_attn shape same as x_tagged. attention shape (bs, heads, entities, entities)
        x_attn = x_attn.permute(0,2,1) # shape bs x chans x (h*w)
        # x_attn = x_attn.unsqueeze(0) # shape 1 x chans x (h*w)
        x_attn = self.feature_wise_maxpool(x_attn) # shape bs x chans x 1
        x_attn = x_attn.squeeze(-1)

        x_attn_mlp = self.mlp_aft_feat_wise_maxpool(x_attn)

        if self.use_memory:
            NotImplementedError
            # hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            # hidden = self.memory_rnn(x_attn, hidden)
            # embedding = hidden[0]
            # memory = torch.cat(hidden, dim=1)
        else:
            embedding = x_attn_mlp

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        if self.use_memory:
            return dist, value, memory
        else:
            return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]



class ACModel_Plain(nn.Module, torch_rl.RecurrentACModel):
    '''
    Actor-critic model for distributional shift env from Deepmind's AI Safety Gridworld.
    Can also use the original "ACModel" class for this env.
    '''
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.recurrent = use_memory

        # Define image embedding
        image_chans = obs_space["image"][2]
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_chans, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU()#,
            # nn.Conv2d(32, 64, (2, 2)),
            # nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64   # original. image_embedding_size is basically the number of elements at output of self.image_conv(x), not accounting for batch size.
        self.image_embedding_size = 32 * 6 # 32 is outchan, 6 is h*w

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 16),
                nn.Tanh(),
                nn.Linear(16, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        if self.use_memory:
            return dist, value, memory
        else:
            return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]






class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]




