'''
The technique you're describing involves using an autoencoder to learn latent representations of images in an 
unsupervised manner, followed by fine-tuning a portion of the network (typically the encoder) with a small amount
of labeled data for a supervised task. This approach is a form of semi-supervised learning and leverages both 
unsupervised pretraining and supervised fine-tuning, making efficient use of both unlabeled and labeled data. 
Specifically, it falls under the category of representation learning followed by transfer learning.
'''
import numpy as np
