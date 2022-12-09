import os
import random
import zipfile
import numpy as np
from util import utils
import tensorflow as tf
from scipy import ndimage
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers

NormalScans = [
    os.path.join(os.getcwd(), "./CT-0", x)
    for x in os.listdir('./CT-0/')
]

AbnormalScans = [
    os.path.join(os.getcwd(), "./CT-23", x)
    for x in os.listdir("./CT-23")
]




NumABOrNormalScans = "CT scans with normal lung tissue: {}\nCT scans with abnormal lung tissue: {}".format(len(NormalScans), len(AbnormalScans))
print(NumABOrNormalScans)

normal_scans = np.array([utils.read_and_process(path) for path in NormalScans])
ab_normal_scans = np.array([utils.read_and_process(path) for path in AbnormalScans])


# For the CT scans having presence of viral pneumonia assign 1, for the normal ones assign 0.
ab_normal_labels = np.array([1 for _ in range(len(ab_normal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data in the ratio 80-20 for training and validation.
x_train = np.concatenate((ab_normal_scans[:80], normal_scans[:80]), axis=0)
y_train = np.concatenate((ab_normal_labels[:80], normal_labels[:80]), axis=0)
x_val = np.concatenate((ab_normal_scans[80:], normal_scans[80:]), axis=0)
y_val = np.concatenate((ab_normal_labels[80:], normal_labels[80:]), axis=0)
PerTrVal = "Number of samples in train and validation are {} and {}.".format(x_train.shape[0], x_val.shape[0])
print(PerTrVal)

@tf.function
def rotate(ctimg):
    def scipy_rotate(ctimg):
        degrees = [-20, -10, -5, 5, 10, 20]
        degree = random.choice(degrees)
        ctimg = ndimage.rotate(ctimg, degree, reshape=False)
        ctimg[ctimg < 0] = 0
        ctimg[ctimg > 1] = 1
        return ctimg
    augmented_ctimg = tf.numpy_function(scipy_rotate, [ctimg], tf.float32)
    return augmented_ctimg

def train_preprocessing(volume, label):
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
batch_size = 2


train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")



def plot_slices(num_rows, num_columns, width, height, data):
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


plot_slices(4, 10, 128, 128, image[:, :, :40])

def get_model(width=128, height=128, depth=64):

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_model(width=128, height=128, depth=64)
model.summary()


initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )






