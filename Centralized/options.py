import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'cifar10',
        help = 'name of the dataset: mnist, cifar10, isic'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'cnn',
        help='name of model. mnist: logistic, lenet; cifar10: cnn_tutorial, cnn_complex; isic: cnn'
    )
    parser.add_argument(
        '--input_channels',
        type = int,
        default = 3,
        help = 'input channels. mnist:1, cifar10:3, isic:3'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 10,
        help = 'output channels (number of classes). mnist:10, cifar10:10, isic:9'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 10,
        help = 'batch size when trained on client'
    )
    parser.add_argument(
        '--num_communication',
        type = int,
        default=1,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=1,
        help='number of local update (tau_1)'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=1,
        help = 'number of edge aggregation (tau_2)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.001,
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '1',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 0,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    #setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = 1,
        help = 'Data distribution type: 1 (IID), 0 (Non-IID balanced), -1 (Non-IID unbalanced), -2 (One-class)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=1,
        help='Edge data distribution type: 1 (IID), 0 (Non-IID) (only used when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1.0,
        help = 'Fraction of clients to use (between 0 and 1)'
    )
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 1,
        help = 'Number of clients for data distribution (default is 1 for centralized learning)'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 1,
        help= 'Number of edge servers (default: 1)'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )
    parser.add_argument(
        '--dataset_root',
        type = str,
        default = 'data',
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--isic_dirname',
        type = str,
        default = 'Skin cancer ISIC The International Skin Imaging Collaboration',
        help = 'ISIC dataset directory name under dataset_root'
    )
    parser.add_argument(
        '--isic_image_size',
        type = int,
        default = 224,
        help = 'size to resize ISIC dataset images (default: 224)'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 0,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default = 2,
        help='Number of classes per client for Non-IID distribution (default: 2)'
    )
    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )

    parser.add_argument(
        '--use_imagenet_stats',
        type = int,
        default = 1,
        help = 'whether to use ImageNet normalization stats (1) or calculate dataset-specific stats (0)'
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args