import numpy as np
import foolbox
import keras
import os

from sklearn.mixture import GaussianMixture
from keras.applications import vgg16, vgg19, resnet50


from keras.applications.resnet50 import ResNet50, decode_predictions

def apply_modifications(model):
    """Credit: keras-vis"""
    import tempfile
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return keras.models.load_model(model_path)
    finally:
        os.remove(model_path)

def B(image, n=1):
    return np.array([image]*n)

def load(path):
    return cv2.resize(cv2.imread(path), (224,224)).astype(np.float32)
        
## instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
kmodel.layers[-1].activation=keras.activations.linear
kmodel=apply_modifications(kmodel)
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

## GG

# get source image and label

image, label = foolbox.utils.imagenet_example()
label=282
image = image[:,:,::-1]
pp = resnet50.preprocess_input
pp = lambda x:x

dist = 98.73439
## Noising
def sample_spherical(npoints, ndim):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec



## CEM

def calculate_margins(images, model, label):
    pred = model.predict(images)
    masked = pred.copy()
    masked[:, label]=0
    close_idx = masked.argmax(1)
    margin = pred[range(len(images)),label]-pred[range(len(images)),close_idx]
    return margin


def cem(image, model, label, n_components=3, n_samples=50, elite_rate=.1,
        n_steps = 1, dist=dist):
    H,W,C = image.shape
    image_base = np.expand_dims(image,0).repeat(n_samples, 0)
    noise = sample_spherical(n_samples, H*W*C).reshape([n_samples,H,W,C])*dist
    samples = image_base + noise
    n_elite = int(elite_rate*n_samples)
    assert(n_elite > 0)
    avg_margins = []
    avg_elite_margins = []
    for step in range(n_steps):
        margin = calculate_margins(samples, model, label)
        avg_margins.append(margin.mean())
        elite_idx = margin.argsort()[:n_elite]
        avg_elite_margins.append(margin[elite_idx].mean())
        elite_samples = samples[elite_idx]
        gm = GaussianMixture(n_components, 'spherical')
        gm.fit(elite_samples.reshape((n_elite, -1)))
        samples = gm.sample(n_samples)[0].reshape(n_samples, H,W,C)
    return gm, avg_margins, avg_elite_margins

class CEM:

    def __init__(self, model, image, label, n_samples=1000,
                 elite_rate=.1, f=GaussianMixture(100,'spherical')):
        H,W,C = image.shape
        self.shape = image.shape
        image_base = np.expand_dims(image,0).repeat(n_samples, 0)
        noise = sample_spherical(n_samples, H*W*C).reshape([n_samples,H,W,C])*dist
        self.samples = image_base + noise
        self.n_samples = n_samples
        self.n_elite = int(elite_rate*n_samples)
        assert(self.n_elite > 0)
        self.avg_margins = []
        self.avg_elite_margins = []
        self.f = f
        self.model = model
        self.label = label

    def step(self):
        margin = calculate_margins(self.samples, self.model, self.label)
        self.avg_margins.append(margin.mean())
        elite_idx = margin.argsort()[:self.n_elite]
        self.avg_elite_margins.append(margin[elite_idx].mean())
        elite_samples = self.samples[elite_idx]
        self.f.fit(elite_samples.reshape((self.n_elite, -1)))
        H,W,C = self.shape
        self.samples = self.f.sample(self.n_samples)[0].reshape(self.n_samples, H,W,C)

