import torch
import matplotlib.pyplot as plt


def smooth(x):
    return torch.nn.functional.avg_pool2d(x, kernel_size=11, stride=1, padding=5)


sigma = 0.05

img = torch.load('ct_image_pytorch.pth')

# normalize image
img = torch.clamp(img+500, 0, 1200)/1200
# img_min = img.min()
# img_max = img.max()
# img -= img_min
# img /= img_max

# bilateral grid
a = torch.linspace(0, 1, 16).view(1, -1, 1, 1)
w = torch.exp(- (a-img).pow(2) / (2*sigma**2))
I_A = w * img
I_result = smooth(smooth(I_A)) / smooth(smooth(w))

# gather output image from the maximum weights
result = torch.gather(I_result, dim=1, index=torch.argmax(w, dim=1, keepdim=True)).squeeze()

# restore intensities
result = result * 1200 - 500
# result = result * img_max + img_min
plt.imshow(result, cmap='gray')
plt.show()
torch.save(result, 'img_denoise_bonus.pth')


