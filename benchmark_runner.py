# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import argparse

from tensorflow.keras.losses import CategoricalCrossentropy, Reduction

from deel.influenciae.benchmark.influence_factory import (
    TracInFactory,
    RPSLJEFactory,
    FirstOrderFactory,
    RPSL2Factory,
    WeightsBoundaryCalculatorFactory,
    SampleBoundaryCalculatorFactory,
    ArnoldiCalculatorFactory
)
from deel.influenciae.benchmark.cifar10_benchmark import Cifar10MislabelingDetectorEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("-epochs", default=60, type=int, help="Number of epochs to train the model")
    parser.add_argument("-model_type", default='resnet', type=str, choices=['resnet', 'efficient_net', 'vgg19'],
                        help="Type of model")
    parser.add_argument("-mislabeling_ratio", default=0.0005, type=float,
                        help="The ratio of samples mislabeled in the dataset")
    parser.add_argument("-use_regu", default=False, type=bool, help="Regularization of the last layers with L1L2 regu")
    parser.add_argument("-force_overfit", default=False, type=bool,
                        help="Force overfiting of the classifier with sgd optimizer")
    parser.add_argument("-train_batch_size", default=128, type=int, help="The batch size used for the training")
    parser.add_argument("-test_batch_size", default=128, type=int,
                        help="The batch size used for the test accuracy metric")
    parser.add_argument("-influence_batch_size", default=128, type=int,
                        help="The batch size used to compute influence functions")

    parser.add_argument("-epochs_to_save", default="",
                        type=lambda x: [int(x_) for x_ in x.split(',')] if len(x) > 0 else None,
                        help="the model used for the tracin method")
    parser.add_argument("-verbose_training", default=False, type=bool,
                        help="Display in the console information about intermediate training steps for each model")
    parser.add_argument("-use_tensorboard", default=False, type=bool, help="Log training data in a tensorboard")

    # Evaluation parameters
    parser.add_argument("-path_to_save", default='./results/', type=str,
                        help="Path to save the result of the benchmark")
    parser.add_argument("-nbr_of_evaluation", default=10, type=int, help="Nbr of seeds used to bench a method")

    parser.add_argument("-method_name", default='influence_first_order', metavar=str, help="methods to benchmark",
                        choices=['influence_first_order', 'tracin', 'rps_lje', 'rps_l2', 'boundary_weights',
                                 'boundary_sample', 'arnoldi'], required=True)

    # Methods parameters
    parser.add_argument("-ihvp_mode", default='exact', type=str, help="Inverse hessian product computation method",
                        choices=['exact', 'cgd', 'lissa'])
    parser.add_argument("-start_layer", default=-1, type=int,
                        help="Starting layer index for the weights and bias collection.")
    parser.add_argument("-dataset_hessian_size", default=2000, type=int,
                        help="The number of samples used for hessian expectation estimation")
    parser.add_argument("-n_opt_iters", default=100, type=int, help="Number of iterations for the optimizer")
    parser.add_argument("-feature_extractor", default=-1, type=int, help="End layer index for the feature extractor")
    parser.add_argument("-lambda_regularization", default=1E-4, type=float, help="L2 regularization for rps L2")
    parser.add_argument("-scaling_factor", default=0.1, type=float, help="Scaling factor for rps L2")
    parser.add_argument("-layer_index", default=-2, type=int, help="Layer index for rps L2")
    parser.add_argument("-epochs_rpsl2", default=100, type=int, help="Epochs to train the dense layer for rps L2")
    parser.add_argument("-boundary_iter", default=100, type=int, help="Number of iterations to found the boundary")
    parser.add_argument("-subspace_dim", default=200, type=int, help="Arnoldi method - subspace projection")
    parser.add_argument("-k_largest_eig_vals", default=100, type=int,
                        help="Arnoldi method - number of top eigenvalues to keep")
    parser.add_argument("-force_hermitian", default=False, type=bool,
                        help="Arnoldi method - force matrix to be hermitian before eigenvalue computation")

    parser.add_argument("-take_batch", default=-1, type=int, help="For debug, keep only a part of the dataset")

    args = parser.parse_args()

    use_bias = False if args.method_name == "rps_lje" or args.method_name == "rps_l2" else True

    cifar10_evaluator = Cifar10MislabelingDetectorEvaluator(epochs=args.epochs,
                                                            model_type=args.model_type,
                                                            mislabeling_ratio=args.mislabeling_ratio,
                                                            use_regu=args.use_regu,
                                                            use_bias=use_bias,
                                                            force_overfit=args.force_overfit,
                                                            train_batch_size=args.train_batch_size,
                                                            test_batch_size=args.test_batch_size,
                                                            influence_batch_size=args.influence_batch_size,
                                                            epochs_to_save=args.epochs_to_save,
                                                            take_batch=args.take_batch,
                                                            verbose_training=args.verbose_training,
                                                            use_tensorboard=args.use_tensorboard)

    if isinstance(args.method_name, str):
        args.method_name = [args.method_name]

    influence_methods_dict = {
        'influence_first_order': FirstOrderFactory(ihvp_mode=args.ihvp_mode,
                                                   start_layer=args.start_layer,
                                                   dataset_hessian_size=args.dataset_hessian_size,
                                                   n_opt_iters=args.n_opt_iters,
                                                   feature_extractor=args.feature_extractor),
        'tracin': TracInFactory(),
        'rps_l2': RPSL2Factory(CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE),
                               args.lambda_regularization,
                               args.scaling_factor,
                               args.layer_index,
                               args.epochs_rpsl2),
        'rps_lje': RPSLJEFactory(ihvp_mode=args.ihvp_mode,
                                 start_layer=args.start_layer,
                                 dataset_hessian_size=args.dataset_hessian_size,
                                 n_opt_iters=args.n_opt_iters,
                                 feature_extractor=args.feature_extractor
                                 ),
        'boundary_weights': WeightsBoundaryCalculatorFactory(step_nbr=args.boundary_iter),
        'boundary_sample': SampleBoundaryCalculatorFactory(step_nbr=args.boundary_iter),
        'arnoldi': ArnoldiCalculatorFactory(subspace_dim=args.subspace_dim,
                                            force_hermitian=args.force_hermitian,
                                            k_largest_eig_vals=args.k_largest_eig_vals,
                                            start_layer=args.start_layer,
                                            dataset_hessian_size=args.dataset_hessian_size)
    }

    factories = {}
    for method_name in args.method_name:
        factories[method_name] = influence_methods_dict[method_name]

    result = cifar10_evaluator.bench(influence_calculator_factories=factories,
                                     nbr_of_evaluation=args.nbr_of_evaluation,
                                     verbose=True,
                                     path_to_save=args.path_to_save,
                                     use_tensorboard=args.use_tensorboard,
                                     seed=0)

    print(result)
