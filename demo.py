import os
import random

import PIL
import cv2 as cv
import numpy as np
import torch
from PIL.Image import Image
import datasets
from models.AE import AE
from device_utils import get_freeish_gpu
from torchvision import transforms
import glob
from utils import resize128_to_tensor, to_PIL_resize1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # checkpoint = '{}/'.format(save_folder)  # model checkpoint
    checkpoint = os.path.join('results', 'checkpoint_10.pt')
    test_csv_path = '/data/matan/nih/test_fixed_path_Data_Entry_2017_v2020.csv'
    pacemakers = glob.glob('/data/matan/nih/pacemakers/*.png')
    image_store = 'images'
    if not os.path.exists(checkpoint):
        exit(1)

    print('checkpoint: ', checkpoint)

    if not os.path.exists(image_store):
        os.mkdir(image_store)

    # Load model
    checkpoint = torch.load(checkpoint)
    #a = AE(dataset=datasets.XRAY(1), log_interval=1, is_cuda=True, batch_size=1)

    a = AE(dataset='XRAY', is_cuda=True, batch_size=128, log_interval=10)
    a.model.load_state_dict(checkpoint)
    #model = a.model.to(get_freeish_gpu())
    a.model.eval()

    import pandas
    df = pandas.read_csv(test_csv_path)
    paths = df['Image Index'].tolist()
    base_paths = [os.path.basename(p) for p in paths]
    test_pacemakers = [p for p in pacemakers if os.path.basename(p) in base_paths]

    img_path = test_pacemakers[0]

    img = PIL.Image.open(img_path).convert('L')

    transfrom = resize128_to_tensor()
    transformed = transfrom(img)
    transformed = transformed[None, ...]
    transformed = transformed.to(a.device)
    img.save(os.path.join(image_store, 'raw_{}'.format(os.path.basename(img_path))))

    preds = a.model(transformed)
    preds = preds.cpu()
    preds = torch.squeeze(preds, 0)
    retransform = to_PIL_resize1024()
    retransformed = retransform(preds)

    retransformed.save(os.path.join(image_store, 'ae_{}'.format(os.path.basename(img_path))))

    from skimage.measure import compare_ssim
    import imutils
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    im1 = np.array(img)
    im2 = np.array(retransformed)

    (score, diff) = compare_ssim(im1, im2, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images

    fig, ax = plt.subplots(2, 2)

    ax[0][0].imshow(im1)
    ax[0][0].set_title('original')

    ax[0][1].imshow(im2)
    ax[0][1].set_title('autoencoder')

    ax[1][0].imshow(diff)
    ax[1][0].set_title('diff')

    ax[1][1].imshow(thresh)
    ax[1][1].set_title('thresh')

    plt.savefig(os.path.join(image_store, 'ae_diff_{}'.format(os.path.basename(img_path))))
    plt.close(fig)
    # for i in range(num_test_samples):
    #     out = preds[i]
    #     out = out.cpu().numpy()
    #     out = np.transpose(out, (1, 2, 0))
    #     out = out * 255.
    #     out = np.clip(out, 0, 255)
    #     out = out.astype(np.uint8)
    #     out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    #     cv.imwrite('images/{}_out.png'.format(i), out)


if __name__ == '__main__':
    with torch.no_grad():
        main()
