import torch.nn as nn
from tdcnn_model import TDCNN


class FacialGestureClassifier(TDCNN):
    def __init__(self, filter_widths, num_classes, num_features, dropout=0.25, channels=64, fc_layers=None):
        super().__init__(filter_widths, num_features, dropout, channels)

        self.fc_layers = fc_layers
        self.num_classes = num_classes
        self.shrink_net = nn.Conv1d(channels, num_classes, 1)
        self.softmax = nn.Softmax()
        self.gesture_net = self.create_gesture_net()
        self.eval()

    def gesture_net(self):
        gesture_net = nn.Sequential()
        i = 1
        input_dim = self.num_classes
        if len(self.fc_layers) > 0:
            h = self.fc_layers[0]
            gesture_net.add_module(f'fc{i}', nn.Linear(input_dim, h))
            gesture_net.add_module(f'relu{i}', nn.ReLU())
            prev_h = h
            i += 1
            for h in self.fc_layers[1:]:
                gesture_net.add_module(f'fc{i}', nn.Linear(prev_h, h))
                gesture_net.add_module(f'relu{i}', nn.ReLU())
                prev_h = h
                i += 1
        else:
            prev_h = self.num_classes
        gesture_net.add_module(f'fc{i}', nn.Linear(prev_h, self.num_classes))
        return gesture_net

    def forward(self, x):
        features = super().forward(x)
        logits = self.shrink_net(features)
        averaged_logits = logits.mean(dim=2)
        gesture_pred = self.softmax(self.gesture_net(averaged_logits))
        return features, averaged_logits, gesture_pred
