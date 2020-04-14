import numpy as np
import os
import glob
from sklearn.cluster import KMeans

# additional library for histogram display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


def build_vocabulary(image_paths, vocab_size):
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))

    # Initialize an array of features, which will store the sampled descriptors
    # keypoints = np.zeros((n_image * n_each, 2))
    descriptors = np.zeros((n_image * n_each, 128))

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)
        sift_keypoints = features[:, :2]
        sift_descriptors = features[:, 2:]

        # TODO: Randomly sample n_each descriptors from sift_descriptor and store them into descriptors
        ind = np.random.choice(sift_descriptors.shape[0], size=min(n_each,sift_descriptors.shape[0]), replace=False)
        descriptors[i*n_each:(i+1)*n_each] = sift_descriptors[ind]
    # TODO: pefrom k-means clustering to cluster sampled sift descriptors into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=vocab_size).fit(descriptors)
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans):
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size))

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Assign each feature to the closest cluster center
        # Again, each feature consists of the (x, y) location and the 128-dimensional sift descriptor
        # You can access the sift descriptors part by features[:, 2:]
        image_feats[i] += np.bincount(kmeans.predict(features[:, 2:]),minlength=vocab_size)
        # TODO: Build a histogram normalized by the number of descriptors
        image_feats[i] /= np.sum(image_feats[i, :])
    return image_feats

def make_avg_histogram(labels, image_feats, categories):
    """ Makes, plots and saves the average histogram
    Parameters
    ----------
    labels: an array of length n_image. Containing class labels correstonding to each image
    image_feats: an 2-dim matrix of size (n_image, vocab_size), where each row is a histogram.
    bags: an array of length 15, storing the names for each label
    """
    # Initialize the histogram
    avg_histogram = np.zeros((15, image_feats.shape[1]))
    # Initialize an bag dictionary
    bags, sizes = np.unique(labels, return_counts=True)
    for i, label in enumerate(labels):
        ind = np.where(bags == label)
        avg_histogram[ind] += image_feats[i]

    # average the histogram
    for i in range(15):
        avg_histogram[i] /= sizes[i]

    # plot and save
    for i in range(15):
        plt.bar(np.arange(len(avg_histogram[i])), avg_histogram[i])
        plt.title('Average historgram for: ' + str(categories[i]))
        plt.savefig(('avg_histograms/avg_histogram_' + str(categories[i])) + '.png', format='png')
        plt.show()



def load(ds_path):
    """ Load from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    labels: class labels corresponding to each image
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)
    image_paths = np.asarray(files)
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_files)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    # Randomize the order
    idx = np.random.choice(n_files, size=n_files, replace=False)
    image_paths = image_paths[idx]
    labels = labels[idx]

    return image_paths, labels

def make_confusion_matrix(y_true, y_pred, title=None, categories=None):
    '''
    Parameters:
        ---------
        y_true: the true labels
        y_pred: the predicted labels
        title: title of the plot
        categories: a list of labels' names
    Return:
        ------
    Returns a plot of confusion matrix
    Code from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    and
    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    '''
    #Compute condusion matrix
    cm = confusion_matrix(y_true, y_pred)
    #normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    accuracy = (sum(cm.diagonal())/cm.sum())*100

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           #Label them with the respective list entries
           xticklabels=categories, yticklabels=categories,
           title=title+', Accuracy: ' + str(round(accuracy,1)) + '%',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.1f'),
                    ha="center", va="center")
    fig.tight_layout()

    plt.savefig(title + '.png', format="png", dpi=300)
    plt.show()


    return ax



if __name__ == "__main__":
    paths, labels = load("sift/train")
    #build_vocabulary(paths, 10)
