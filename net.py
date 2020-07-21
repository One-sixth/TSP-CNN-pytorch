import torch
import torch.nn as nn
import numpy as np


class TSPCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        y = self.layers(x)
        return y

    # def load_from_my_w(self, p):
    #     ws = np.load(p, allow_pickle=True)['ws']
    #     params = list(self.layers.parameters())
    #     for i, w in enumerate(ws):
    #         if i % 2 == 0:
    #             w = np.transpose(w, [3, 2, 0, 1])
    #             # w = np.transpose(w, [0, 1, 3, 2])
    #         # elif i % 2 == 1:
    #         #     w = w[::-1].copy()
    #         params[i].data[:] = torch.tensor(w)
    #         print(w.shape)


if __name__ == '__main__':
    import imageio
    import cv2

    torch.set_grad_enabled(False)
    net = TSPCNN()

    # net.load_from_my_w('w.npz')
    # torch.save(net.state_dict(), 'tsp_cnn.pt')

    net.load_state_dict(torch.load('tsp_cnn.pt', map_location='cpu'))

    net.eval()

    im = imageio.imread('a.bmp')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.asarray(im, np.float32) / 255
    im = im[None, None]
    im = torch.tensor(im)
    hm = net(im).numpy()[0, 0]
    hm = np.clip(hm * 255., 0, 255).astype(np.uint8)
    imageio.imwrite('b.bmp', hm)
