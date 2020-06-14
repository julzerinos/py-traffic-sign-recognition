import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.compat.v1.layers import flatten
from pipeline import NeuralNetwork, make_adam, Session, build_pipeline

tf.compat.v1.disable_eager_execution()

matplotlib.style.use('ggplot')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

TRAIN_IMAGE_DIR = 'data/Final_Training/Images'

dfs = []
for train_file in glob.glob(os.path.join(TRAIN_IMAGE_DIR, '*/GT-*.csv')):
    folder = train_file.split('/')[3]
    df = pd.read_csv(train_file, sep=';')
    df['Filename'] = df['Filename'].apply(lambda x: os.path.join(TRAIN_IMAGE_DIR, folder, x))
    dfs.append(df)

train_df = pd.concat(dfs, ignore_index=True)
train_df.head()

N_CLASSES = np.unique(train_df['ClassId']).size  # keep this for later

print("Number of training images : {:>5}".format(train_df.shape[0]))
print("Number of classes         : {:>5}".format(N_CLASSES))



def show_class_distribution(classIDs, title):
    """
    Plot the traffic sign class distribution
    """
    plt.figure(figsize=(15, 5))
    plt.title('Class ID distribution for {}'.format(title))
    plt.hist(classIDs, bins=N_CLASSES)
    plt.show()



show_class_distribution(train_df['ClassId'], 'Train Data')


sign_name_df = pd.read_csv('sign_names.csv', index_col='ClassId')
sign_name_df.head()


sign_name_df['Occurence'] = [sum(train_df['ClassId']==c) for c in range(N_CLASSES)]
sign_name_df.sort_values('Occurence', ascending=False)


SIGN_NAMES = sign_name_df.SignName.values
SIGN_NAMES[2]


def load_image(image_file):
    """
    Read image file into numpy array (RGB)
    """
    return plt.imread(image_file)


def get_samples(image_data, num_samples, class_id=None):
    """
    Randomly select image filenames and their class IDs
    """
    if class_id is not None:
        image_data = image_data[image_data['ClassId']==class_id]
    indices = np.random.choice(image_data.shape[0], size=num_samples, replace=False)
    return image_data.iloc[indices][['Filename', 'ClassId']].values


def show_images(image_data, cols=5, sign_names=None, show_shape=False, func=None):
    """
    Given a list of image file paths, load images and show them.
    """
    num_images = len(image_data)
    rows = num_images//cols
    plt.figure(figsize=(cols*3,rows*2.5))
    for i, (image_file, label) in enumerate(image_data):
        image = load_image(image_file)
        if func is not None:
            image = func(image)
        plt.subplot(rows, cols, i+1)
        plt.imshow(image)
        if sign_names is not None:
            plt.text(0, 0, '{}: {}'.format(label, sign_names[label]), color='k',backgroundcolor='c', fontsize=8)
        if show_shape:
            plt.text(0, image.shape[0], '{}'.format(image.shape), color='k',backgroundcolor='y', fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()


sample_data = get_samples(train_df, 20)
show_images(sample_data, sign_names=SIGN_NAMES, show_shape=True)




print(SIGN_NAMES[2])
show_images(get_samples(train_df, 100, class_id=2), cols=20, show_shape=True)




X = train_df['Filename'].values
y = train_df['ClassId'].values

print('X data', len(X))


X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=8000, random_state=0)

print('X_train:', len(X_train))
print('X_valid:', len(X_valid))


INPUT_SHAPE = (32, 32, 3)

def make_network1(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 6])
            .max_pool()
            .relu()
            .conv([5, 5, 16])
            .max_pool()
            .relu()
            .flatten()
            .dense(120)
            .relu()
            .dense(N_CLASSES))


def train_evaluate(pipeline, epochs=5, samples_per_epoch=50000, train=(X_train, y_train), test=(X_valid, y_valid)):
    """
    Repeat the training for the epochs and evaluate the performance
    """
    X, y = train
    learning_curve = []
    for i in range(epochs):
        indices = np.random.choice(len(X), size=samples_per_epoch)
        pipeline.fit(X[indices], y[indices])
        scores = [pipeline.score(*train), pipeline.score(*test)]
        learning_curve.append([i, *scores])
        print("Epoch: {:>3} Train Score: {:.3f} Evaluation Score: {:.3f}".format(i, *scores))
    return np.array(learning_curve).T # (epochs, train scores, eval scores)


def resize_image(image, shape=INPUT_SHAPE[:2]):
    return cv2.resize(image, shape)

loader = lambda image_file: resize_image(load_image(image_file))


with Session() as session:
    functions = [loader]
    pipeline = build_pipeline(functions, session, make_network1(), make_adam(1.0e-3))
    train_evaluate(pipeline)


def random_brightness(image, ratio):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = np.float64(hsv[:, :, 2])
    brightness = brightness * (1.0 + np.random.uniform(-ratio, ratio))
    brightness[brightness>255] = 255
    brightness[brightness<0] = 0
    hsv[:, :, 2] = brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_rotation(image, angle):
    """
    Randomly rotate the image
    """
    if angle == 0:
        return image
    angle = np.random.uniform(-angle, angle)
    rows, cols = image.shape[:2]
    size = cols, rows
    center = cols/2, rows/2
    scale = 1.0
    rotation = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, rotation, size)


def random_translation(image, translation):
    """
    Randomly move the image
    """
    if translation == 0:
        return 0
    rows, cols = image.shape[:2]
    size = cols, rows
    x = np.random.uniform(-translation, translation)
    y = np.random.uniform(-translation, translation)
    trans = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image, trans, size)


def random_shear(image, shear):
    """
    Randomly distort the image
    """
    if shear == 0:
        return image
    rows, cols = image.shape[:2]
    size = cols, rows
    left, right, top, bottom = shear, cols - shear, shear, rows - shear
    dx = np.random.uniform(-shear, shear)
    dy = np.random.uniform(-shear, shear)
    p1 = np.float32([[left   , top],[right   , top   ],[left, bottom]])
    p2 = np.float32([[left+dx, top],[right+dx, top+dy],[left, bottom+dy]])
    move = cv2.getAffineTransform(p1,p2)
    return cv2.warpAffine(image, move, size)


def augment_image(image, brightness, angle, translation, shear):
    image = random_brightness(image, brightness)
    image = random_rotation(image, angle)
    image = random_translation(image, translation)
    image = random_shear(image, shear)
    return image


augmenter = lambda x: augment_image(x, brightness=0.7, angle=10, translation=5, shear=2)

show_images(sample_data[10:], cols=10) # original
for _ in range(5):
    show_images(sample_data[10:], cols=10, func=augmenter)


with Session() as session:
    functions = [loader, augmenter]
    pipeline = build_pipeline(functions, session, make_network1(), make_adam(1.0e-3))
    train_evaluate(pipeline)


normalizers = [('x - 127.5',              lambda x: x - 127.5),
               ('x/127.5 - 1.0',          lambda x: x/127.5 - 1.0),
               ('x/255.0 - 0.5',          lambda x: x/255.0 - 0.5),
               ('x - x.mean()',           lambda x: x - x.mean()),
               ('(x - x.mean())/x.std()', lambda x: (x - x.mean())/x.std())]

for name, normalizer in normalizers:
    print('Normalizer: {}'.format(name))
    with Session() as session:
        functions = [loader, augmenter, normalizer]
        pipeline = build_pipeline(functions, session, make_network1(), make_adam(1.0e-3))
        train_evaluate(pipeline)
    print()


normalizer = lambda x: (x - x.mean())/x.std()


# for Gray scale, we need to add the 3rd dimension back (1 channel) as it's expected by the network
converters = [('Gray', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]),
              ('HSV', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV)),
              ('HLS', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HLS)),
              ('Lab', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2Lab)),
              ('Luv', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2Luv)),
              ('XYZ', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2XYZ)),
              ('Yrb', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)),
              ('YUV', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YUV))]

GRAY_INPUT_SHAPE = (*INPUT_SHAPE[:2], 1)

for name, converter in converters:
    print('Color Space: {}'.format(name))
    with Session() as session:
        functions = [loader, augmenter, converter, normalizer]
        if name == 'Gray':
            network = make_network1(input_shape=GRAY_INPUT_SHAPE) # there is only one channel in gray scale
        else:
            network = make_network1()
        pipeline = build_pipeline(functions, session, network, make_adam(1.0e-3))
        train_evaluate(pipeline)
    print()

preprocessors = [loader, augmenter, normalizer]




def show_learning_curve(learning_curve):
    epochs, train, valid = learning_curve
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, train, label='train')
    plt.plot(epochs, valid, label='validation')
    plt.title('Learning Curve')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.xticks(epochs)
    plt.legend(loc='center right')

def plot_confusion_matrix(cm):
    cm = [row/sum(row)   for row in cm]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Oranges)
    fig.colorbar(cax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class IDs')
    plt.ylabel('True Class IDs')
    plt.show()


def print_confusion_matrix(cm, sign_names=SIGN_NAMES):
    results = [(i, SIGN_NAMES[i], row[i]/sum(row)*100) for i, row in enumerate(cm)]
    accuracies = []
    for result in sorted(results, key=lambda x: -x[2]):
        print('{:>2} {:<50} {:6.2f}% {:>4}'.format(*result, sum(y_train==result[0])))
        accuracies.append(result[2])
    print('-'*50)
    print('Accuracy: Mean: {:.3f} Std: {:.3f}'.format(np.mean(accuracies), np.std(accuracies)))

def make_network2(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 12])  # <== doubled
            .max_pool()
            .relu()
            .conv([5, 5, 32])  # <== doubled
            .max_pool()
            .relu()
            .flatten()
            .dense(240) # <== doubled
            .relu()
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network2(), make_adam(1.0e-3))
    learning_curve = train_evaluate(pipeline)
    session.save('checkpoint/network2.ckpt')

show_learning_curve(learning_curve)

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network2())
    session.load('checkpoint/network2.ckpt')
    pred = pipeline.predict(X_valid)

# examine confusionconfusion_matrixix
cm = confusion_matrix(y_valid, pred)
plot_confusion_matrix(cm)
print_confusion_matrix(cm)


def make_network3(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24]) # <== doubled
            .max_pool()
            .relu()
            .conv([5, 5, 64]) # <== doubled
            .max_pool()
            .relu()
            .flatten()
            .dense(480)  # <== doubled
            .relu()
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3(), make_adam(1.0e-3))
    learning_curve = train_evaluate(pipeline)
    session.save('checkpoint/network3.ckpt')

show_learning_curve(learning_curve)


with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3())
    session.load('checkpoint/network3.ckpt')
    pred = pipeline.predict(X_valid)

# examine confusionconfusion_matrixix
cm = confusion_matrix(y_valid, pred)
plot_confusion_matrix(cm)
print_confusion_matrix(cm)




with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3(), make_adam(1.0e-3))
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network3_epochs-20.ckpt')

show_learning_curve(learning_curve)





with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3())
    session.load('checkpoint/network3_epochs-20.ckpt')
    pred = pipeline.predict(X_valid)

# examine confusionconfusion_matrixix
cm = confusion_matrix(y_valid, pred)
plot_confusion_matrix(cm)
print_confusion_matrix(cm)


with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3(), make_adam(0.5e-3)) # <== lower learning rate
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network3_epochs-20_lr-0.5e-3.ckpt')

show_learning_curve(learning_curve)


with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3())
    session.load('checkpoint/network3_epochs-20_lr-0.5e-3.ckpt')
    pred = pipeline.predict(X_valid)

# examine confusionconfusion_matrixix
cm = confusion_matrix(y_valid, pred)
plot_confusion_matrix(cm)
print_confusion_matrix(cm)


with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3(), make_adam(1.0e-4)) # <== lower learning rate
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network3_epochs-20_lr-1.0e-4.ckpt')

show_learning_curve(learning_curve)


with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3())
    session.load('checkpoint/network3_epochs-20_lr-1.0e-4.ckpt')
    pred = pipeline.predict(X_valid)

# examine confusionconfusion_matrixix
cm = confusion_matrix(y_valid, pred)
plot_confusion_matrix(cm)
print_confusion_matrix(cm)


def make_network4(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24])
            .max_pool()
            .relu(leak_ratio=0.01) # <== leaky ReLU
            .conv([5, 5, 64])
            .max_pool()
            .relu(leak_ratio=0.01) # <== leaky ReLU
            .flatten()
            .dense(480)
            .relu(leak_ratio=0.01) # <== leaky ReLU
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network4(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network4.ckpt')

show_learning_curve(learning_curve)




def make_network5(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24])
            .max_pool()
            .elu()              # <== ELU
            .conv([5, 5, 64])
            .max_pool()
            .elu()              # <== ELU
            .flatten()
            .dense(480)
            .elu()              # <== ELU
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network5(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network5.ckpt')

show_learning_curve(learning_curve)


def make_network6(input_shape=INPUT_SHAPE):
    return (NeuralNetwork(weight_sigma=0.01) # <== smaller weight sigma
            .input(input_shape)
            .conv([5, 5, 24])
            .max_pool()
            .relu()
            .conv([5, 5, 64])
            .max_pool()
            .relu()
            .flatten()
            .dense(480)
            .relu()
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network6(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network6.ckpt')

show_learning_curve(learning_curve)


def make_network7(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24])
            .max_pool()
            .relu()
            .conv([5, 5, 64])
            .max_pool()
            .relu()
            .flatten()
            .dense(480)
            .relu()
            .dense(240) # <== one more dense layer
            .relu()
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network7(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network7.ckpt')

show_learning_curve(learning_curve)

def make_network8(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24])
            .relu()
            .max_pool() # <== after ReLU
            .conv([5, 5, 64])
            .relu()
            .max_pool() # <== after ReLU
            .flatten()
            .dense(480)
            .relu()
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network8(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network8.ckpt')

show_learning_curve(learning_curve)


def make_network9(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24])
            .max_pool()
            .relu()
            .conv([5, 5, 64])
            .max_pool()
            .relu()
            .conv([3, 3, 64])  # <= smaller kernel here (the image is small by here)
            .max_pool()
            .relu()
            .flatten()
            .dense(480)
            .relu()
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network9(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=20)
    session.save('checkpoint/network9.ckpt')

show_learning_curve(learning_curve)


for momentum in [0.7, 0.8, 0.9]:
    with Session() as session:
        print('Momentum: {}'.format(momentum))
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.5e-3, momentum=momentum)
        pipeline = build_pipeline(preprocessors, session, make_network3(), optimizer)
        train_evaluate(pipeline, epochs=20)
        session.save('checkpoint/network3_momentum_{}.ckpt'.format(momentum))
        print()


def balance_distribution(X, y, size):
    X_balanced = []
    y_balanced = []
    for c in range(N_CLASSES):
        data = X[y==c]
        indices = np.random.choice(sum(y==c), size)
        X_balanced.extend(X[y==c][indices])
        y_balanced.extend(y[y==c][indices])
    return np.array(X_balanced), np.array(y_balanced)

X_balanced, y_balanced = balance_distribution(X_train, y_train, 3000)

show_class_distribution(y_balanced, 'Balanced Train Set')


with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=20, train=(X_balanced, y_balanced)) # <== using the balanced train set
    session.save('checkpoint/network3_with_balanced_data.ckpt')

show_learning_curve(learning_curve)




with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3(), make_adam(0.5e-3))
    learning_curve = train_evaluate(pipeline, epochs=100)

show_learning_curve(learning_curve)





with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3(), make_adam(1.0e-4))
    learning_curve = train_evaluate(pipeline, epochs=500)
    session.save('checkpoint/network3_epochs-500_lr-1.0e-4.ckpt')

show_learning_curve(learning_curve)


def make_network10(input_shape=INPUT_SHAPE):
    return (NeuralNetwork()
            .input(input_shape)
            .conv([5, 5, 24])
            .max_pool()
            .relu()
            .conv([5, 5, 64])
            .max_pool()
            .relu()
            .dropout(keep_prob=0.5)
            .flatten()
            .dense(480)
            .relu()
            .dense(N_CLASSES))

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network10(), make_adam(1.0e-4))
    learning_curve = train_evaluate(pipeline, epochs=500)
    session.save('checkpoint/network10.ckpt')

show_learning_curve(learning_curve)




with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network10())
    session.load('checkpoint/network10.ckpt')
    pred = pipeline.predict(X_valid)

# examine confusionconfusion_matrixix
cm = confusion_matrix(y_valid, pred)
plot_confusion_matrix(cm)
print_confusion_matrix(cm)


def enhance_image(image, ksize, weight):
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return cv2.addWeighted(image, weight, blurred, -weight, image.mean())


for ksize in [5, 7, 9, 11]:
    for weight in [4, 6, 8, 10]:
        print('Enhancer: k={} w={}'.format(ksize, weight))
        with Session() as session:
            enhancer = lambda x: enhance_image(x, ksize, weight)
            functions = [loader, augmenter, enhancer, normalizer]
            pipeline = build_pipeline(functions, session, make_network10())
            session.load('checkpoint/network10.ckpt')
            score = pipeline.score(X_valid, y_valid)
            print('Validation Score: {}'.format(score))
        print()


enhancer = lambda x: enhance_image(x, 9, 8)

show_images(sample_data[10:], cols=10)
show_images(sample_data[10:], cols=10, func=enhancer)


def equalizer(image):
    image = image.copy()
    for i in range(3):
        image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return image


show_images(sample_data[10:], cols=10)
show_images(sample_data[10:], cols=10, func=equalizer)




with Session() as session:
    functions = [loader, augmenter, equalizer, normalizer]
    pipeline = build_pipeline(functions, session, make_network10())
    session.load('checkpoint/network10.ckpt')
    score = pipeline.score(X_valid, y_valid)
    print('Validation Score: {:.3f}'.format(score))




with Session() as session:
    functions = [loader, augmenter, equalizer, enhancer, normalizer]
    pipeline = build_pipeline(functions, session, make_network10())
    session.load('checkpoint/network10.ckpt')
    score = pipeline.score(X_valid, y_valid)
    print(score)




def min_max_norm(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

show_images(sample_data[10:], cols=10)
show_images(sample_data[10:], cols=10, func=min_max_norm)


with Session() as session:
    functions = [loader, augmenter, min_max_norm, normalizer]
    pipeline = build_pipeline(functions, session, make_network10())
    session.load('checkpoint/network10.ckpt')
    score = pipeline.score(X_valid, y_valid)
    print(score)




with Session() as session:
    functions = [loader, augmenter, min_max_norm, enhancer, normalizer]
    pipeline = build_pipeline(functions, session, make_network10())
    session.load('checkpoint/network10.ckpt')
    score = pipeline.score(X_valid, y_valid)
    print(score)


TEST_IMAGE_DIR = 'data/Final_Test/Images'

# Note: GT-final_test.csv comes with class IDs (GT-final_test.test.csv does not)
test_df = pd.read_csv(os.path.join(TEST_IMAGE_DIR, 'GT-final_test.csv'), sep=';')
test_df['Filename'] = test_df['Filename'].apply(lambda x: os.path.join(TEST_IMAGE_DIR, x))
test_df.head()


print("Number of test images: {:>5}".format(test_df.shape[0]))

X_test = test_df['Filename'].values
y_test = test_df['ClassId'].values

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3())
    session.load('checkpoint/network10.ckpt')
    score = pipeline.score(X_test, y_test)
    print('Test Score: {}'.format(score))

X_new = np.array(glob.glob('images/sign*.jpg') +
                 glob.glob('images/sign*.png'))

new_images = [plt.imread(path) for path in X_new]


print('-' * 80)
print('New Images for Random Testing')
print('-' * 80)

plt.figure(figsize=(15,5))
for i, image in enumerate(new_images):
    plt.subplot(2,len(X_new)//2,i+1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()




print('getting top 5 results')

with Session() as session:
    pipeline = build_pipeline(preprocessors, session, make_network3())
    session.load('checkpoint/network10.ckpt')
    prob = pipeline.predict_proba(X_new)
    estimator = pipeline.steps[-1][1]
    top_5_prob, top_5_pred = estimator.top_k_

print('done')

print('-' * 80)
print('Top 5 Predictions')
print('-' * 80)

for i, (preds, probs, image) in enumerate(zip(top_5_pred, top_5_prob, new_images)):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    for pred, prob in zip(preds.astype(int), probs):
        sign_name = SIGN_NAMES[pred]
        print('{:>5}: {:<50} ({:>14.10f}%)'.format(pred, sign_name, prob*100.0))
    print('-' * 80)
