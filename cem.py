import numpy as np
import foolbox
import keras


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
print('original', kmodel.predict(pp(np.array([image]))).argmax())
# apply attack on source image
attack = foolbox.attacks.GradientAttack(fmodel)
# attack = foolbox.attacks.GradientSignAttack(fmodel)
# attack = foolbox.attacks.DeepFoolAttack(fmodel)
adversarial = attack(image, label)
print('adversarial', kmodel.predict(pp(np.array([adversarial]))).argmax())
dist = np.linalg.norm(adversarial-image)
dist = 98.73439
## Noising
def sample_spherical(npoints, ndim):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


## Sample points
N = 100

noise = sample_spherical(N,224**2*3).reshape([N]+list(image.shape))*dist
batch = np.array([image]*N)
noised = noise+batch
pred = kmodel.predict(noised)
print('adversarial', pred.argmax(1))

true_score = pred.argmax(1)
masked = pred.copy()
masked[:,label]=0
margin = (pred[:,label])-np.max(masked,1)
print('margin',margin)

print('min m', np.min(margin))
print('Predicted:', decode_predictions(pred, top=3)[0])

## GM
gm = GaussianMixture(3, 'spherical')
gm.fit(noised.reshape([N,-1]))


## CEM

def calculate_margins(images, model, label):
    pred = model.predict(images)
    masked = pred.copy()
    masked[:, label]=0
    close_idx = masked.argmax(1)
    margin = pred[range(len(images)),label]-pred[range(len(images)),close_idx]
    return margin
def CEM(image, model, label, n_components=3, n_samples=50, elite_rate=.1,
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