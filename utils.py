import numpy as np
import torch
from skimage import io, transform


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks) if type(landmarks) is not str else landmarks}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
#        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


def get_interpolations(args, model, device, images, images_per_row=20):
    model.eval()
    with torch.no_grad():
        def interpolate(t1, t2, num_interps):
            alpha = np.linspace(0, 1, num_interps+2)
            interps = []
            for a in alpha:
                interps.append(a*t2.view(1, -1) + (1 - a)*t1.view(1, -1))
            return torch.cat(interps, 0)

        if args.model == 'VAE':
            mu, logvar = model.encode(images.view(-1, 784))
            embeddings = model.reparameterize(mu, logvar).cpu()
        elif args.model == 'AE':
            embeddings = model.encode(images.view(-1, 784))
            
        interps = []
        for i in range(0, images_per_row+1, 1):
            interp = interpolate(embeddings[i], embeddings[i+1], images_per_row-4)
            interp = interp.to(device)
            interp_dec = model.decode(interp)
            line = torch.cat((images[i].view(-1, 784), interp_dec, images[i+1].view(-1, 784)))
            interps.append(line)
        # Complete the loop and append the first image again
        interp = interpolate(embeddings[i+1], embeddings[0], images_per_row-4)
        interp = interp.to(device)
        interp_dec = model.decode(interp)
        line = torch.cat((images[i+1].view(-1, 784), interp_dec, images[0].view(-1, 784)))
        interps.append(line)

        interps = torch.cat(interps, 0).to(device)
    return interps
