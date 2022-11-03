from deel.influenciae.benchmark.influence_factory import TracInFactory, RPSLJEFactory, FirstOrderFactory, RPSL2Factory
from deel.influenciae.benchmark.cifar10_benchmark import Cifar10MislabelingDetectorEvaluator
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("-epochs", default=60, type=int, help="Number of epochs to train the model")
    parser.add_argument("-model_type", default='resnet', type=str, choices=['resnet', 'efficient_net', 'vgg19'],
                        help="Type of model")
    parser.add_argument("-mislabeling_ratio", default=0.0005, type=float,
                        help="The ratio of samples miss labeled in the dataset")
    parser.add_argument("-use_regu", default=False, type=bool, help="Regularization of the last layers with L1L2 regu")
    parser.add_argument("-force_overfit", default=False, type=bool,
                        help="Force overfiting of the classifier with sgd optimizer")
    parser.add_argument("-train_batch_size", default=128, type=int, help="The batch size used for the training")
    parser.add_argument("-test_batch_size", default=128, type=int,
                        help="The batch size used for the test accuracy metric")
    parser.add_argument("-influence_batch_size", default=128, type=int,
                        help="The batch size used to compute influence function")

    parser.add_argument("-epochs_to_save", default="20, 40, 60", type=lambda x: [int(x_) for x_ in x.split(',')],
                        help="the model used for the tracin method")
    parser.add_argument("-verbose_training", default=False, type=bool,
                        help="Display in the console intermediate traning step for each models")
    parser.add_argument("-use_tensorboard", default=False, type=bool, help="Log training data in a tensorboard")

    # Evaluation parameters
    parser.add_argument("-path_to_save", default='./results/', type=str,
                        help="Path to save the result of the benchmark")
    parser.add_argument("-nbr_of_evaluation", default=10, type=int, help="Nbr of seed used to bench a method")

    parser.add_argument("-method_name", default='influence_first_order', metavar=str, help="methods ot benchmark",
                        choices=['influence_first_order', 'tracein', 'rps_lje', 'rps_l2'], required=True)

    # Methods parameters
    parser.add_argument("-ihvp_mode", default='exact', type=str, help="Inverse hessian product computation method",
                        choices=['exact', 'cgd'])
    parser.add_argument("-start_layer", default=-1, type=int,
                        help="Starting layer index for the weights and bias collection.")
    parser.add_argument("-dataset_hessian_size", default=2000, type=int,
                        help="The number of sample used for hessian esperance estimation")
    parser.add_argument("-n_cgd_iters", default=100, type=int, help="Number of iterations for cgd")
    parser.add_argument("-feature_extractor", default=-1, type=int, help="End layer index for the feature extractor")
    parser.add_argument("-lambda_regularization", default=1E-4, type=float, help="L2 regularization for rps L2")
    parser.add_argument("-scaling_factor", default=0.1, type=float, help="Scaling factor for rps L2")
    parser.add_argument("-layer_index", default=-2, type=int, help="Layer index for rps L2")
    parser.add_argument("-epochs_rpsl2", default=100, type=int, help="Epochs to train the dense layer for rps L2")

    parser.add_argument("-take_batch", default=-1, type=int, help="For debug, keep only a part of the dataset")

    args = parser.parse_args()

    cifar10_evaluator = Cifar10MislabelingDetectorEvaluator(epochs=args.epochs,
                                                            model_type=args.model_type,
                                                            mislabeling_ratio=args.mislabeling_ratio,
                                                            use_regu=args.use_regu,
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

    factories = {}
    for method_name in args.method_name:
        if method_name == 'influence_first_order':
            influence_factory = FirstOrderFactory(ihvp_mode=args.ihvp_mode,
                                                  start_layer=args.start_layer,
                                                  dataset_hessian_size=args.dataset_hessian_size,
                                                  n_cgd_iters=args.n_cgd_iters,
                                                  feature_extractor=args.feature_extractor)

        elif method_name == 'tracein':
            influence_factory = TracInFactory()
        elif method_name == 'rps_l2':
            influence_factory = RPSL2Factory(args.lambda_regularization,
                                             args.scaling_factor,
                                             args.layer_index,
                                             args.epochs_rpsl2)
        elif method_name == 'rps_lje':
            influence_factory = RPSLJEFactory(ihvp_mode=args.ihvp_mode,
                                              start_layer=args.start_layer,
                                              dataset_hessian_size=args.dataset_hessian_size,
                                              n_cgd_iters=args.n_cgd_iters,
                                              feature_extractor=args.feature_extractor
                                              )
        else:
            raise Exception('Unknown method to benchmark=' + method_name)

        factories[method_name] = influence_factory

    result = cifar10_evaluator.bench(influence_calculator_factories=factories,
                                     nbr_of_evaluation=args.nbr_of_evaluation,
                                     verbose=True,
                                     path_to_save=args.path_to_save,
                                     use_tensorboard=args.use_tensorboard,
                                     seed=0)

    print(result)
