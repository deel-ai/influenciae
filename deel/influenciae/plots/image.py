# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module implementing plotting functions for image-type data.
"""
from math import ceil

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ..types import Optional, Union


def _normalize(image: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].

    Parameters
    ----------
    image
        Image to prepare.

    Returns
    -------
    image
        Image ready to be used with matplotlib (in range[0, 1]).
    """
    image = np.array(image, np.float32)

    image -= image.min()
    image /= image.max()

    return image


def plot_most_influential_images(
        influential_images: tf.data.Dataset,
        cols: int = 5,
        img_size: float = 2.,
        save_path: Optional[str] = None
):
    """
    Plots the collection of images in a grid.

    Parameters
    ----------
    influential_images
        A dataset with the influential images and their influence scores.
    cols
        An integer indicating the amount of columns for the plot.
    img_size
        Size of each subplots (in inches), considering we keep aspect ratio.
    save_path
        An optional string specifying the path where to save the plot. If None, the figure
        will be displayed directly on screen.
    """
    # Materialize the contents of the dataset with the influential images
    images, influence_values = list(influential_images)

    # Calculate the amount of rows needed for this plot
    rows = ceil(len(images) / cols)
    # get width and height of our images
    l_width, l_height = images.shape[1:-1]

    # define the figure margin, width, height in inch
    margin = 0.3
    spacing = 0.3
    figwidth = cols * img_size + (cols - 1) * spacing + 2 * margin
    figheight = rows * img_size * l_height / l_width + (rows - 1) * spacing + 2 * margin

    left = margin / figwidth
    bottom = margin / figheight

    fig = plt.figure()
    fig.set_size_inches(figwidth, figheight)

    fig.subplots_adjust(
        left=left,
        bottom=bottom,
        right=1. - left,
        top=1. - bottom,
        wspace=spacing / img_size,
        hspace=spacing / img_size * l_width / l_height
    )

    if influence_values is not None:
        for i, (image, value) in enumerate(zip(images, influence_values)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(_normalize(image))
            plt.axis('off')
            plt.gca().set_title(f'inf_value = {value:.2f}')
    else:
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(_normalize(image))
            plt.axis('off')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def plot_datacentric_explanations(
        image: Union[tf.Tensor, np.ndarray],
        explanations: tf.data.Dataset,
        cols: int = 5,
        img_size: float = 2.,
        save_path: Optional[str] = None
):
    """
    Plots the most influential points for a test image, serving as a 'data-centric'
    explanation for why the model predicted a certain way.

    Parameters
    ----------
    image
        A tensor or numpy array with the image we wish to explain.
    explanations
        A dataset with the top-k most influential images for the provided image and their
        corresponding influence values.
    cols
        An integer indicating the amount of columns for the plot.
    img_size
        Size of each subplots (in inches), considering we keep aspect ratio.
    save_path
        An optional string specifying the path where to save the plot. If None, the figure
        will be displayed directly on screen.
    """
    # Calculate the amount of rows needed for this plot
    rows = ceil(len(explanations) / cols) + 1
    # get width and height of our images
    l_width, l_height = image.shape[:-1]

    # define the figure margin, width, height in inch
    margin = 0.3
    spacing = 0.3
    figwidth = cols * img_size + (cols - 1) * spacing + 2 * margin
    figheight = rows * img_size * l_height / l_width + (rows - 1) * spacing + 2 * margin

    left = margin / figwidth
    bottom = margin / figheight

    fig = plt.figure()
    fig.set_size_inches(figwidth, figheight)

    fig.subplots_adjust(
        left=left,
        bottom=bottom,
        right=1. - left,
        top=1. - bottom,
        wspace=spacing / img_size,
        hspace=spacing / img_size * l_width / l_height
    )

    plt.subplot(rows + 1, cols, 1)
    plt.imshow(_normalize(image))
    plt.axis('off')
    for i, (top_k_image, top_k_inf_value) in enumerate(zip(*explanations), start=1):
        plt.subplot(rows + 1, cols, i + cols)
        plt.imshow(_normalize(top_k_image))
        plt.axis('off')
        plt.gca().set_title(f'inf_value = {top_k_inf_value:.2f}')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
