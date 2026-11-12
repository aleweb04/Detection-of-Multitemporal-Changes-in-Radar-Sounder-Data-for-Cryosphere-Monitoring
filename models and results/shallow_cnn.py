import torch
import torch.nn as nn


# TODO: INTEGRARE RETE [ N >= L layers, L=4 ( L-1 convoluzionali  conv 3x3 con stride=1 i.e. no stride , 1 layer di proiezione lineare i.e. con layer size 1×1) ]
#  CHE NON ABBIA STRATI DI POOLING ( NO MAX. POOLING ) i.e. CHE NON ABBIA RIDUZIONE DI DIMENSIONALITÀ SEGUITA DA UPSAMPLING
#  ( i.e. features nei diversi layers tutte con la stessa dimensione )
#  e.g. blocchi unitari rete : conv, Relu, BN ]

# FIXME: COME STRUTTURARE LA RETE per poter estrarre le features da L <= N layers interni in fase di TESTING?? (TUTORIALS SU ESTRAZIONE FEATURES DA LAYERS INTERNI)
import models.builder_moco as bld 
#from models.builder_moco import bld.conv_block


class ShallowCNN(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=5,
            depth=5,    # TODO: usare tale parametro per definire il # di conv layers della rete [ e.g. ciclo for su range(1,depth) ]
            norm_layer=None,
    ):
        super(ShallowCNN, self).__init__()
        # assert up_mode in ('upconv', 'upsample')
        self.depth = depth

        # TODO: refactor usando il parametro DEPTH per definire il # di conv layers della rete
        #  e.g. from UNet code
        #  and using setattr(self, 'Conv'+str(i), value) to set layers in un loop
        self.Conv1 = bld.conv_block(ch_in=in_channels, ch_out=8, norm_layer=norm_layer)
        self.Conv2 = bld.conv_block(ch_in=8, ch_out=16, norm_layer=norm_layer)
        self.Conv3 = bld.conv_block(ch_in=16, ch_out=32, norm_layer=norm_layer)
        self.Conv4 = bld.conv_block(ch_in=32, ch_out=64, norm_layer=norm_layer)

        self.last = nn.Sequential(
            # TODO: add optional DROPOUT layer
            nn.Conv2d(64, n_classes, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            # nn.ReLU(inplace=True)
            norm_layer(num_features=n_classes)
        )

    def forward(self, x):

        # TODO: refactor usando il parametro DEPTH per definire il # di conv layers della rete
        print("ShallowCNN x: ", x.shape)
        x1 = self.Conv1(x)
        print("ShallowCNN x1: ", x1.shape)
        x2 = self.Conv2(x1)
        print("ShallowCNN x2: ", x2.shape)
        x3 = self.Conv3(x2)
        print("ShallowCNN x3: ", x3.shape)
        x4 = self.Conv4(x3)
        print("ShallowCNN x4: ", x4.shape)
        last = self.last(x4)
        print("ShallowCNN last: ", last.shape)
        return last








