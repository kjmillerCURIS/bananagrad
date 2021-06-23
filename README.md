# bananagrad
Using backprop to confuse a cat classifier with a banana!

A fun way to familiarize myself with PyTorch and the inner workings of its backprop. It uses a custom "PhotoshopFunction" function (i.e. torch.autograd.Function) to paste a foreground onto a background. This function is differentiable w.r.t. the (x, y) location where the foreground is pasted, which allows us to do fun stuff like trying to fool a cat classifier by strategically pasting a banana over the cat.

    python bananagrad.py cat.jpeg banana.png vis
