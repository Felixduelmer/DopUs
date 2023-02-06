import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj, mkdirs
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from utils.early_stopping import EarlyStopping
import pandas as pd

from models import get_model


def train(arguments):
    # Parse input arguments
    json_filename = arguments.config

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    save_dir = os.path.join(
        json_opts.model.checkpoints_dir, json_opts.model.experiment_name)

    mkdirs([save_dir])
    shutil.copy(json_filename, save_dir)

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(
        arch_type, opts=json_opts.augmentation)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=save_dir)
    error_logger = ErrorLogger()

    for fold in range(7):

        scores = {}

        # Setup the NN Model
        model = get_model(json_opts.model)
        # Setup Data Loader
        train_dataset = ds_class(ds_path, split='train', fold=fold,
                                 preload_data=train_opts.preloadData, transform=ds_transform['train'])
        test_dataset = ds_class(ds_path, split='test', fold=fold,
                                preload_data=train_opts.preloadData)
        train_loader = DataLoader(
            dataset=train_dataset, num_workers=0, batch_size=train_opts.batchSize, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, num_workers=0,
                                 batch_size=train_opts.batchSize, shuffle=True)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=train_opts.patience, verbose=True)

        # Training Function
        model.set_scheduler(train_opts)
        for epoch in range(model.which_epoch, train_opts.n_epochs):
            print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

            # Training Iterations
            for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
                # Make a training update
                model.init_hidden(images.size(0), images.size(3))

                for i in range(train_opts.seq_len):
                    model.set_input(images[:, i, :, :, :], labels[:, i, :, :, :])
                    model.forward('train')

                    # Optimize the parameters only if 1) model is not recurrent 2) the bptt step size is reached
                    # 3) the sequence is about to end
                    if not json_opts.model.is_rnn or ((i + 1) % train_opts.update_freq) == 0 or (
                            i + 1) == train_opts.seq_len:
                        model.optimize_parameters()
                        errors = model.get_current_errors()
                        stats = model.get_segmentation_stats()
                        error_logger.update({**errors, **stats}, split='train')

            # Validation and Testing Iterations
            for epoch_iter, (images, labels) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
                # Make a forward pass with the model
                model.init_hidden(images.size(0), images.size(3))
                for i in range(train_opts.seq_len):
                    model.set_input(images[:, i, :, :, :],
                                    labels[:, i, :, :, :])
                    model.validate()

                    # Error visualisation
                    errors = model.get_current_errors()
                    stats = model.get_segmentation_stats()
                    error_logger.update({**errors, **stats}, split='test')

                    # Visualise predictions

                    visuals = model.get_current_visuals(labels[:, i, [0], :, :])
                    visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

            # Update the plots
            for split in ['train', 'test']:
                visualizer.plot_current_errors(
                    epoch, error_logger.get_errors(split), split_name=split + '_fold_' + str(fold))
                visualizer.print_current_errors(
                    epoch, error_logger.get_errors(split), split_name=split + '_fold_' + str(fold))

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(error_logger.get_errors('test').get('Seg_Loss'))
            tmp_dict = error_logger.get_errors('test')
            tmp_dict['seg_loss_train'] = error_logger.get_errors('train').get('Seg_Loss')

            for key in tmp_dict:
                if key in scores:
                    scores[key].append(tmp_dict[key])
                else:
                    scores[key] = [tmp_dict[key]]

            # Save the model parameters
            if epoch % train_opts.save_epoch_freq == 0:
                model.save_fold(epoch, fold)

            # Update the model learning rate
            model.update_learning_rate()
            error_logger.reset()

            # save to file
            df = pd.DataFrame(scores)
            df.to_csv(save_dir + '/' + str(fold) + '.csv')

            if early_stopping.early_stop or epoch is train_opts.n_epochs - 1:
                print("Stopping due to no improvement or max epochs has been reached")
                break
        del model
        del early_stopping
        del train_loader, train_dataset
        del test_loader, test_dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Seg Training Function')

    parser.add_argument(
        '-c', '--config', help='training config file', required=True)
    args = parser.parse_args()

    train(args)
