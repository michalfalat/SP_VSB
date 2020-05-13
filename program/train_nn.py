from nn import run_train, run_test, visualise_model
import argparse
"""Train neural network file"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    """main"""
    parser = argparse.ArgumentParser(description='Program for analysing driver\'s behaviour in vehicle')
    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--model-name', dest='modelName', action='store', help='Final name of trained model', default='defaultModel.keras')
    optional.add_argument('--path', dest='path', action='store', help='Path to folder  with train images grouped by folders', default='./program/train_nn/')
    optional.add_argument('--epochs', dest='epochs', action='store', type=int, help='Number of epochs', default=12)
    args = parser.parse_args()
    run_train(args)
    run_test(args)
    visualise_model(args)

main()
