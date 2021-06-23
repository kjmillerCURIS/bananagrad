import os
import sys
import cv2
import numpy as np
from scipy.ndimage import sobel
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Function
import torchvision.models as models
from torchvision import transforms

#copy-paste a banana onto an image to mess with an existing classifier
#try to optimize the mess-up-ed-ness w.r.t. the (x, y) location of the banana
#how to do it:
# I = alpha * I_banana + (1 - alpha) * I_orig
# d alpha / d (x, y) = -spatial_gradient(alpha)
# d I_banana / d (x, y) = -spatial_gradient(I_banana)
# d I / d (x, y) = -alpha * spatial_gradient(I_banana) - (I_banana - I_orig) * spatial_gradient(alpha)
#Pretty sure that's what Lucas-Kanade says...
#Can do some really simple inpainting to get background of banana image so spatial gradient isn't crazy
#The sigmas for the spatial gradients can be parameters
#Although it's not exactly equivalent to just using a blurry banana
#unless you actually do the forward pass with a blurry banana

CONSTANT_HEIGHT = 224
CAT_INDICES = [281,282,283,284,285]
EPSILON = 1e-5 #to keep the loss from blowing up when Pr(cat) == 1
BANANA_SIGMA = 4.0
MASK_SIGMA = 4.0
SCALE = 0.5
#LR = 1.5e+1
LR = 1e+1
NUM_ITERS = 500
INIT_XY = (50, 50)

#sanity-check
#the "banana" is a white square, and the "cat" is a color gradient
#the "classifier" looks at the sum of the image
#so the (x, y) gradient should point in roughly the same direction as the color gradient
#because moving the "banana" in that direction would make the composite image maximally brighter/darker
def stuff():
    photoshopAndClassify = PhotoshopAndClassify(2.0, 2.0, use_sum_clf=True)
    photoshopAndClassify.to('cuda')
    xMesh, yMesh = np.meshgrid(-np.arange(224), -np.arange(224))
    xMesh = xMesh / 224.0
    yMesh = yMesh / 224.0

    #the image gets darker in the (0.31, 0.69) direction
    #therefore, we should have d sum(Xcomposite) / d xy = K * (0.31, 0.69) for some K > 0.0
    #that's because moving the "banana" to the darker part makes the image brighter
    meow = 1.0 - 0.31 * xMesh - 0.69 * yMesh #gradient direction should be (-0.31, -0.69)
    
    
    xOrig = torch.from_numpy(np.repeat(meow.astype('float32')[np.newaxis, np.newaxis, :, :], 3, axis=1)).to('cuda')
    xBanana = torch.from_numpy(np.ones((1, 3, 224, 224), dtype='float32')).to('cuda')
    xMask = np.ones((1, 1, 224, 224), dtype='float32')
    xMask[:,:,80:120,80:120] = 1.0
    xMask = torch.from_numpy(xMask).to('cuda')
    mrow = photoshopAndClassify(xOrig, xBanana, xMask)
    mrow.backward()
    print(photoshopAndClassify.xy.grad)
    print(photoshopAndClassify.xy.grad / torch.sum(torch.abs(photoshopAndClassify.xy.grad)))

    import pdb
    pdb.set_trace()

#you should only need to call save_for_backward() in here, NOT anything else special!!!
class PhotoshopFunction(Function):

    @staticmethod
    #just does the math of making the filter
    #returns a 2D filter, which should have odd dimensions and be square
    #x_or_y should be 'x' or 'y'
    def __make_spatial_derivative_onefilter_justmath(sigma, x_or_y):
        #sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        #therefore, ksize=2*((sigma-0.8)/0.3+1)+1
        ksize = 2 * ((sigma - 0.8) / 0.3 + 1) + 1
        ksize = int(round(ksize))
        if ksize % 2 == 0:
            ksize += 1

        diffs1D = np.arange(ksize).astype('float32') - ksize // 2
        sqDiffs2D = np.square(diffs1D[np.newaxis,:]) + np.square(diffs1D[:,np.newaxis])
        myFilter = np.exp(-sqDiffs2D / (2.0 * sigma ** 2)) / (2.0 * np.pi * sigma ** 2)
        if x_or_y == 'x':
            myFilter = sobel(myFilter, axis=1)
        elif x_or_y == 'y':
            myFilter = sobel(myFilter, axis=0)
        else:
            assert(False)

        return myFilter

    @staticmethod
    #x_or_y should be 'x' or 'y'
    def __make_spatial_derivative_onefilter(sigma, num_channels, x_or_y):
        myFilterNpy = PhotoshopFunction.__make_spatial_derivative_onefilter_justmath(sigma, x_or_y)
        assert(len(myFilterNpy.shape) == 2)
        assert(myFilterNpy.shape[0] == myFilterNpy.shape[1])
        assert(myFilterNpy.shape[0] % 2 == 1)
        kernel_size = myFilterNpy.shape[0]
        myFilterNpy = np.repeat(myFilterNpy[np.newaxis, np.newaxis, :, :], num_channels, axis=0)
        return torch.from_numpy(myFilterNpy)

    @staticmethod
    #will return dictionary of nn.Conv2d objects
    def make_spatial_derivative_filters(banana_sigma, mask_sigma):
        d = {}
        for s, sigma, num_channels in zip(['Banana', 'Mask'], [banana_sigma, mask_sigma], [3, 1]):
            for x_or_y in ['x', 'y']:
                d[x_or_y + 'Filter' + s] = PhotoshopFunction.__make_spatial_derivative_onefilter(sigma, num_channels, x_or_y)

        return d

    @staticmethod
    #should for for 1 channel or 3
    #will be applied to "natural" images *and* masks *and* spatial gradients
    #this will do wrapping cuz that's the easiest to implement
    #I'm *pretty* sure if I do wrapping for the composite and wrapping of the spatial derivs in backward() then I've got everything correct
    #Like, as long as the *original* banana isn't close to the edge, then everything's fine
    #and we can pretend we live in a world where all images are circular and wrap around :)
    #yep, let's do that!
    def __shift_image(Ximg, xy):
        return torch.roll(Ximg, shifts=torch.round(xy).type(torch.IntTensor).tolist(), dims=(3,2))

    @staticmethod
    #see PhotoshopAndClassify.forward(), PhotoshopAndClassify.xy
    #spatial_derivative_filters should be a dictionary
    def forward(ctx, Xorig, Xbanana, Xmask, xy, spatial_derivative_filters):
        ctx.save_for_backward(*([Xorig, Xbanana, Xmask, xy] + [spatial_derivative_filters[k] for k in ['xFilterBanana', 'yFilterBanana', 'xFilterMask', 'yFilterMask']]))
        Xbanana = PhotoshopFunction.__shift_image(Xbanana, xy)
        Xmask = PhotoshopFunction.__shift_image(Xmask, xy)
        Xcomposite = Xmask * Xbanana + (1.0 - Xmask) * Xorig
        return Xcomposite

    @staticmethod
    #should work for 1 channel or 3
    #Note the lack of ctx - I don't need it here!
    #xFilter and yFilter should be Sobel combined with Gaussian
    #their channels should match the channels of Ximg
    #will return xDeriv, yDeriv
    def __get_spatial_derivative(Ximg, xFilter, yFilter):
        xConv = torch.nn.Conv2d(Ximg.shape[1], Ximg.shape[1], xFilter.shape[2], groups=Ximg.shape[1], bias=False, padding=(xFilter.shape[2]-1)//2)
        xConv.weight.data = xFilter
        xConv.weight.requires_grad=False
        yConv = torch.nn.Conv2d(Ximg.shape[1], Ximg.shape[1], yFilter.shape[2], groups=Ximg.shape[1], bias=False, padding=(yFilter.shape[2]-1)//2)
        yConv.weight.data = yFilter
        yConv.weight.requires_grad=False
        return xConv(Ximg), yConv(Ximg)

    @staticmethod
    #just one output gradient - the composite image!
    # d I / d (x, y) = -alpha * spatial_gradient(I_banana) - (I_banana - I_orig) * spatial_gradient(alpha)
    def backward(ctx, grad_Xcomposite):
        Xorig, Xbanana, Xmask, xy, xFilterBanana, yFilterBanana, xFilterMask, yFilterMask = ctx.saved_tensors
        xDerivBanana, yDerivBanana = PhotoshopFunction.__get_spatial_derivative(Xbanana, xFilterBanana, yFilterBanana)
        xDerivMask, yDerivMask = PhotoshopFunction.__get_spatial_derivative(Xmask, xFilterMask, yFilterMask)
        Xbanana = PhotoshopFunction.__shift_image(Xbanana, xy)
        Xmask = PhotoshopFunction.__shift_image(Xmask, xy)
        xDerivBanana = PhotoshopFunction.__shift_image(xDerivBanana, xy)
        yDerivBanana = PhotoshopFunction.__shift_image(yDerivBanana, xy)
        xDerivMask = PhotoshopFunction.__shift_image(xDerivMask, xy)
        yDerivMask = PhotoshopFunction.__shift_image(yDerivMask, xy)
        xGrad = torch.sum(grad_Xcomposite*(-Xmask * xDerivBanana - (Xbanana - Xorig) * xDerivMask))
        yGrad = torch.sum(grad_Xcomposite*(-Xmask * yDerivBanana - (Xbanana - Xorig) * yDerivMask))
        grad_xy = torch.cat((xGrad.unsqueeze(0), yGrad.unsqueeze(0)))
        return None, None, None, grad_xy, None

class PhotoshopAndClassify(nn.Module):
    def __init__(self, banana_sigma, mask_sigma, use_sum_clf=False):
        super(PhotoshopAndClassify, self).__init__()
        self.spatial_derivative_filters = PhotoshopFunction.make_spatial_derivative_filters(banana_sigma, mask_sigma)
        if use_sum_clf:
            self.clf = torch.sum
        else:
            self.clf = models.vgg16(pretrained=True)

        #this will cause self.xy to come into existence(!)
        self.register_parameter(name='xy', param=nn.Parameter(torch.from_numpy(np.array(INIT_XY, dtype='float32')).requires_grad_()))

    def __make_visualization(self, Xcomposite, clf_logits):
        assert(clf_logits.shape[0] == 1)
        clf_probs = torch.nn.Softmax(dim=1)(clf_logits)
        cat_prob = torch.sum(clf_probs[0, CAT_INDICES]).item()
        numIcomposite = np.squeeze(Xcomposite.permute([0,2,3,1]).detach().cpu().numpy())
        numIcomposite = np.minimum(np.maximum(np.around(255.0 * numIcomposite[:,:,::-1]), 0), 255).astype('uint8')
        #text is annoying, and resolution is small, so let's just put a "health" bar
        numIbar = np.zeros((50, numIcomposite.shape[1], 3), dtype='uint8')
        numIbar[:45,:min(max(int(round(cat_prob*numIcomposite.shape[1])), 0), numIcomposite.shape[1]), 0] = 255
        return np.vstack((numIbar, numIcomposite))

    #all inputs are expected to be same size, NCHW, 0-1, float32
    #this means Xbanana and Xmask should already be centered or in their "initial" position
    #as self.xy is initialized to (0, 0)
    #will return logits
    #will also return visualization as numpy array (like the kind you could immediately cv2.imwrite()) if make_vis is True
    def forward(self, Xorig, Xbanana, Xmask, make_vis=False):
        Xcomposite = PhotoshopFunction.apply(Xorig, Xbanana, Xmask, self.xy, self.spatial_derivative_filters)
        clf_logits = self.clf(transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])(Xcomposite))
        
        if make_vis:
            numIvis = self.__make_visualization(Xcomposite, clf_logits)
            return clf_logits, numIvis
        else:
            return clf_logits

    def to(self, *cargs, **kwargs):
        super(PhotoshopAndClassify, self).to(*cargs, **kwargs)
        my_device = cargs[0]
        for k in sorted(self.spatial_derivative_filters.keys()):
            #it's okay that it's not a leaf node anymore
            #we won't need the gradient
            self.spatial_derivative_filters[k] = self.spatial_derivative_filters[k].to(my_device)

#returns Xorig, Xbanana, Xmask
def process_images(numIorig, numIbanana, scale):
    #first, resize orig to a constant height that's similar to the size used to train from ImageNet
    #that way the NN will get things at the scale they were trained on
    new_width = int(round(numIorig.shape[1] / numIorig.shape[0] * CONSTANT_HEIGHT))
    numIorig = cv2.resize(numIorig, (new_width, CONSTANT_HEIGHT))

    #tight crop around banana foreground
    [r, c] = np.nonzero(numIbanana[:,:,-1] == 255)
    numIbanana = numIbanana[np.amin(r):np.amax(r)+1, np.amin(c):np.amax(c)+1, :]

    #resize banana so banana_height == scale * orig_height
    new_height = int(round(scale * numIorig.shape[0]))
    new_width = int(round(numIbanana.shape[1] / numIbanana.shape[0] * new_height))
    numIbanana = cv2.resize(numIbanana, (new_width, new_height))

    #either pad or crop banana so that it is same size as orig
    if numIbanana.shape[0] < numIorig.shape[0]:
        diff = numIorig.shape[0] - numIbanana.shape[0]
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        numIbanana = np.vstack((np.zeros((pad_top, numIbanana.shape[1], 4), dtype='uint8'), numIbanana, np.zeros((pad_bottom, numIbanana.shape[1], 4), dtype='uint8')))
    elif numIbanana.shape[0] > numIorig.shape[0]:
        diff = numIbanana.shape[0] - numIorig.shape[0]
        crop_top = diff // 2
        crop_bottom = diff - crop_top
        numIbanana = numIbanana[crop_top:numIbanana.shape[0]-crop_bottom, :, :]

    if numIbanana.shape[1] < numIorig.shape[1]:
        diff = numIorig.shape[1] - numIbanana.shape[1]
        pad_left = diff // 2
        pad_right = diff - pad_left
        numIbanana = np.hstack((np.zeros((numIbanana.shape[0], pad_left, 4), dtype='uint8'), numIbanana, np.zeros((numIbanana.shape[0], pad_right, 4), dtype='uint8')))
    elif numIbanana.shape[1] > numIorig.shape[1]:
        diff = numIbanana.shape[1] - numIorig.shape[1]
        crop_left = diff // 2
        crop_right = diff - crop_left
        numIbanana = numIbanana[:, crop_left:numIbanana.shape[1]-crop_right, :]

    #inpaint background so BGR of banana is smooth
    numIbanana[:,:,:-1] = cv2.inpaint(numIbanana[:,:,:-1], (numIbanana[:,:,-1] < 255).astype('uint8'), 3, cv2.INPAINT_TELEA)

    #make background of banana mask zero
    numIbanana[numIbanana[:,:,-1] < 255,-1] = 0

    #form into tensors
    #note that the classifier is expecting BGR on a 0-1 scale
    Xorig = torch.from_numpy((numIorig[np.newaxis,:,:,[2,1,0]] / 255.0).astype('float32')).permute([0, 3, 1, 2]).to('cuda')
    Xbanana = torch.from_numpy((numIbanana[np.newaxis,:,:,[2,1,0]] / 255.0).astype('float32')).permute([0, 3, 1, 2]).to('cuda')
    Xmask = torch.from_numpy((numIbanana[np.newaxis,:,:,[-1]] / 255.0).astype('float32')).permute([0, 3, 1, 2]).to('cuda')
    return Xorig, Xbanana, Xmask

def bananagrad(orig_image, banana_image, vis_dir):
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    numIorig = cv2.imread(orig_image)
    numIbanana = cv2.imread(banana_image, -1)
    assert(numIbanana.shape[2] == 4)
    vis_prefix = os.path.join(vis_dir, os.path.splitext(os.path.basename(orig_image))[0] + '-' + os.path.splitext(os.path.basename(banana_image))[0])
    Xorig, Xbanana, Xmask = process_images(numIorig, numIbanana, SCALE)
    photoshopAndClassify = PhotoshopAndClassify(BANANA_SIGMA, MASK_SIGMA)
    photoshopAndClassify.to('cuda')
#    optimizer = torch.optim.SGD([photoshopAndClassify.xy], lr=LR)
    optimizer = torch.optim.Adam([photoshopAndClassify.xy], lr=LR)
    for t in tqdm(range(NUM_ITERS)):
        clf_logits, numIvis = photoshopAndClassify(Xorig, Xbanana, Xmask, make_vis=True)
        numIbar = np.zeros((30, numIvis.shape[1], 3), dtype='uint8')
        numIbar[:20,:min(max(int(round(t/NUM_ITERS*numIbar.shape[1])), 0), numIbar.shape[1]), 1:] = 255
        cv2.imwrite(vis_prefix + '-%05d.jpg'%(t), np.vstack((numIbar, numIvis)))
        clf_probs = torch.nn.Softmax(dim=1)(clf_logits)
        cat_prob = torch.sum(clf_probs[:, CAT_INDICES], dim=1, keepdim=True)
        loss = torch.mean(-torch.log(1.0 + EPSILON - cat_prob))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def usage():
    print('Usage: python bananagrad.py <orig_image> <banana_image> <vis_dir>')

if __name__ == '__main__':
    bananagrad(*(sys.argv[1:]))
