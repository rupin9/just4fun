
## https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

import torch
import torchvision.models as models

# retrieve pre-trained model (download)
vgg16_mdl = models.vgg16(pretrained=True)
torch.save(vgg16_mdl.state_dict(), 'vgg16_mdl_weights.pth')

# new vgg16 model (not trained) to reload weights
vgg16_new = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
vgg16_new.load_state_dict(torch.load('vgg16_mdl_weights.pth'))

# Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.
vgg16_new.eval()

# save model class
torch.save(vgg16_new, 'vgg16_new.pth')

# load model class
vgg16_again = torch.load('vgg16_new.pth')

# For furthermore,
# look up: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

