import torch
import torch.functional as F
import torch.nn as nn

TORCH_ACTIVATION_LIST = ['ReLU',
                         'Sigmoid',
                         'SELU',
                         'leaky_relu',
                         'Softplus']

ACTIVATION_LIST = ['mish', 'swish', None]


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def get_nn_activation(activation: 'str'):
    if not activation in TORCH_ACTIVATION_LIST + ACTIVATION_LIST:
        raise RuntimeError("Not implemented activation function!")

    if activation in TORCH_ACTIVATION_LIST:
        act = getattr(nn, activation)()

    if activation in ACTIVATION_LIST:
        if activation == 'mish':
            act = Mish()
        elif activation == 'swish':
            act = Swish()
        elif activation is None:
            act = nn.Identity()

    return act


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=[32, 32],
                 hidden_activation='ReLU',
                 out_activation=None,
                 weight_init='normal'
                 ):

        super(MultiLayerPerceptron, self).__init__()

        input_dims = [input_dim] + hidden_dim
        output_dims = hidden_dim + [output_dim]

        # Input -> the last hidden layer
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims[:-1], output_dims[:-1])):
            linear_module = nn.Linear(in_features=in_dim, out_features=out_dim)
            # initializing weight
            self.apply_weight_init(linear_module, weight_init)

            self.layers.append(linear_module)
            self.activations.append(get_nn_activation(hidden_activation))

        output_layer = nn.Linear(in_features=input_dims[-1], out_features=output_dims[-1])
        self.apply_weight_init(output_layer, weight_init)
        self.layers.append(output_layer)
        self.activations.append(get_nn_activation(out_activation))

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            x = activation(x)
        return x

    def check_input_spec(self, input_spec, hidden_dim):
        if isinstance(input_spec, list):
            # output layer will not be normalized
            assert len(input_spec) == len(hidden_dim) + 1, "the length of input_spec list should " \
                                                           "match with the number of hidden layers + 1"
            _list_type = True
        else:
            _list_type = False

        return _list_type

    def apply_weight_init(self, tensor, weight_init=None):
        if weight_init is None:
            pass  # do not apply weight init
        elif weight_init == "normal":
            nn.init.normal_(tensor.weight, std=0.3)
            nn.init.constant_(tensor.bias, 0.0)
        elif weight_init == "kaiming_normal":
            if self.activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
                nn.init.kaiming_normal_(tensor.weight, nonlinearity=self.activation)
                nn.init.constant_(tensor.bias, 0.0)
            else:
                pass
        elif weight_init == "xavier":
            nn.init.xavier_uniform_(tensor.weight)
            nn.init.constant_(tensor.bias, 0.0)
        else:
            raise NotImplementedError("MLP initializer {} is not supported".format(weight_init))

