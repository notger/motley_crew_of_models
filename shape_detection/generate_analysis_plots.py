"""Script to generate analysis plots on a chosen model."""

import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from shape_generator import ShapeTypes, Colouring
from cnn_model import ShapeDetectorModelCNN
from data_loader import ShapeIterableDataset


# Set some hyperparameters:
# Note that those should be the same as the ones you have chosen for training.
N_x, N_y, N_c, N_target = 50, 50, 3, len(ShapeTypes)
colouring = Colouring.RANDOM_PIXELS
batch_size = 1000
shape_names = [s.name for s in ShapeTypes]
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')


def array_to_image(im_array):
    """Very ugly and unreadably converter function, because I can."""
    return Image.fromarray(
        np.moveaxis(  # Swaps the axis from the tensor-ordering (3, Nx, Ny) to (Nx, Ny, 3).
            (im_array * 255 / im_array.max()).astype(np.uint8),  # Convert and rescale the tensor to get a uint8-array.
            [0, 1, 2],
            [2, 0, 1],
        ),
        mode='RGB'
    )


shape_cnn = ShapeDetectorModelCNN.load_from_checkpoint(
    checkpoint_path='checkpoints/best.ckpt',
    N_c=N_c, N_target=N_target,
)

test_data = ShapeIterableDataset(N_x, N_y, colouring, batch_size=batch_size)

# Generate labels and predictions:
predictions = []
labels = []
images = []
errors = []

shape_cnn.freeze()
for im_tensor, label in tqdm(test_data):
    images.append(im_tensor.numpy())
    yp = shape_cnn(im_tensor.unsqueeze(dim=0)).numpy().squeeze()
    predictions.append(yp.argmax())
    labels.append(label)
    errors.append(1 - yp[label])
shape_cnn.unfreeze()

# Plot the results:
sns.set(rc={"figure.figsize":(15, 10)})
ax = sns.heatmap(
    confusion_matrix(labels, predictions, normalize='true'), 
    annot=True
)
_ = ax.set(
    xlabel='predicted as', 
    ylabel='true label', 
    xticklabels=[s.name for s in ShapeTypes],
    yticklabels=[s.name for s in ShapeTypes]
)
plt.show()
plt.savefig(f'plots/{timestamp}_best.png')

# Plot three of the most egregiously wrong examples:
sorted_error_list_with_indices = sorted(list(zip(errors, list(range(len(errors))))), reverse=True)

for k in range(3):
    _, i = sorted_error_list_with_indices[k]
    pred_label_name = shape_names[predictions[i]]
    true_label_name = shape_names[labels[i]]
    im = array_to_image(images[i])
    print(f'Label {true_label_name} was wrongly predicted as {pred_label_name} (index {i}).')
    #im.show()
    im.save(f'plots/{timestamp}_wrong_pred_{pred_label_name}_for_true_{true_label_name}_{k}.png')

# Plot three good examples as well:
for k in range(3):
    _, i = sorted_error_list_with_indices[-k-1]
    pred_label_name = shape_names[predictions[i]]
    true_label_name = shape_names[labels[i]]
    im = array_to_image(images[i])
    print(f'Label {true_label_name} was correctly predicted as {pred_label_name} (index {i}).')
    #im.show()
    im.save(f'plots/{timestamp}_good_pred_for_true_{true_label_name}_{k}.png')
