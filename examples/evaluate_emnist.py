'''
An Evaluation Script for Graphical Models with likelihood estimation trained
on EMNIST Digits and tested on EMNIST characters.

For tests use in conjuction with a simple DBN model (testing limits things to few data points):

python3 evaluate_emnist.py --directory test --file ModelDBN.py --model ModelDBN --tqdm --maxepochs 5 --testing

'''



import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import site

site.addsitedir('..')

#from dbm import DeepBoltzmannMachine
from itertools import chain
from tqdm import tqdm
from os import mkdir
import json
import argparse
from os.path import basename, dirname, join
from shutil import copyfile
from scipy.integrate import simps
from tqdm import tqdm

device = None

def identity(x):
    return x

def build_roc_graph(model, positive, negative, ais=False):
    '''
    This function returns the TPR and FPR for 100 discriminator values.
    :param model: the model to be tested.
    :param positive: positive examples
    :param negative: negative examples
    :param ais: if the model uses ais to evaluate the probability, set this parameter to True
    :return: FPR [torch.tensor(100, float)], TPR [torch.tensor(100, float)]
    '''


    # For models with AIS, the partitioning sum only has to be evaluated once.
    if ais:
        log_z = model.get_log_Z(1000, 1000)
    points = 100
    TP_R = torch.zeros([points])
    FP_R = torch.zeros([points])
    u_positive = []
    u_negative = []

    # Evaluate the log likelihood for the positive and negative examples.
    if ais:
        for pos in positive:
            u_positive.append(model.loglikelihood(data = pos, log_Z = log_z).cpu())
        for neg in negative:
            u_negative.append(model.loglikelihood(data = neg, log_Z = log_z).cpu())
    else:
        for pos in positive:
            u_positive.append(model.loglikelihood(data = pos).cpu())
        for neg in negative:
            u_negative.append(model.loglikelihood(data = neg).cpu())
    u_positive = torch.cat(u_positive, dim=0)
    u_negative = torch.cat(u_negative, dim=0)

    # Throw away samples with underflow / overflow.
    inf_positive = torch.zeros_like(u_positive)
    inf_positive[torch.isinf(u_positive)] = 1
    u_positive = u_positive[inf_positive == 0]

    inf_negative = torch.zeros_like(u_negative)
    inf_negative[torch.isinf(u_negative)] = 1
    u_negative = u_negative[inf_negative == 0]

    # Calculate the step size for the discriminator.
    min_like = min(torch.min(u_positive), torch.min(u_negative))
    max_like = max(torch.max(u_positive), torch.max(u_negative))
    diff = (max_like - min_like) / points

    # Count, how many samples were classified true positive or false positive.
    for i in range(points):
        TP_R[i] = (u_positive[u_positive >= min_like + i*diff]).size(0) / u_positive.size(0)
        FP_R[i] = (u_negative[u_negative >= min_like + i*diff]).size(0) / u_negative.size(0)
    return TP_R, FP_R


def plot_roc_graph(fig, TP_R, FP_R, directory):
    '''
    This function plots the ROC curve for TPR and FPR with common discriminator values and stores it in the path directory.
    :param fig:
    :param TP_R: TPR for different discriminator values
    :param FP_R: FPR for different discriminator values
    :param directory: directory to save the plot
    :return: ROC curve
    '''

    # Set up the plot.
    plt.clf()
    ax = fig.subplots()
    ax.plot(FP_R, TP_R)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Calculate the Yuoden index
    youden_index_list = TP_R - FP_R
    youden_index = youden_index_list.max()

    # Delete not strictly monotonic values for the integration.
    mask = torch.zeros([TP_R.size(0)], dtype=torch.long)
    for i in range(TP_R.size(0) - 1):
        if FP_R[i] == FP_R[i + 1]:
            mask[i + 1] = 1

    # Integrate the curve.
    integral = simps(TP_R[mask == 0], FP_R[mask == 0])

    # Plot the curve.
    fig.suptitle('Youden index = ' + str(round(float(youden_index), 2)) + '    Integral = ' + str(round(abs(integral), 2)))
    fig.savefig(directory + "/roc.png")

def main():
    description = '''
        Evaluate a hierarchical graphical model on emnist.
        Models are learnt via a train() method, samples are generated using
        generate() and the log likelihood (per sample) is estimated using
        loglikelihood().
        This script takes an file defining the model, the model class name,
        some model arguments and whether it should be trained anew.
        It trains the models using a predefined scheme and uses a validation set
        to be able to stop early.
        Finally a test likelihood is calculated, the model is saved, some samples
        are generated and a discriminator between EMNIST digits and characters
        is evaluated, which is based on the loglikelihood() estimation.
        This is done via an ROC-curve, which is stored as well.
        Note that the likelihood-estimator does not have to be normalized, for
        the model to work also as a discriminator. Just the results might not be
        as interpretable.
        '''

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--directory', dest='directory', type=str,
                        help='directory to store the results',
                        required=True)
    parser.add_argument('--file', dest='file', type=str,
                        help='file to read the model from',
                        required=True)
    parser.add_argument('--model', type=str, dest='model',
                        help='model class name')
    parser.add_argument('--retrain', dest='retrain',
                        const=True, default=False, action='store_const',
                        help='retrain previously trained model')
    parser.add_argument('--ais', dest='ais',
                        const=True, default=False, action='store_const',
                        help='use ais partition sum precalculation')
    parser.add_argument('--minepochs', dest='minepochs', type=int, default=20,
                        help='minimal number of epochs')
    parser.add_argument('--maxepochs', dest='maxepochs', type=int, default=1000,
                        help='maximal number of epochs')
    parser.add_argument('--valepochs', dest='valepochs', type=int, default=5,
                        help='validation loglik after every $ epochs')
    parser.add_argument('--reeval', dest='reeval',
                        const=True, default=False, action='store_const',
                        help='reevaluate loglikelihood on test and valid data')
    parser.add_argument('--tqdm', dest='tqdm',
                        const=tqdm, default=identity, action='store_const',
                        help='use tqdm to show progress')
    parser.add_argument('--no-binarize', dest='binarize',
                        const=False, default=True, action='store_const',
                        help='don\'t binarize EMNIST data')
    parser.add_argument('--copy', dest='copy',
                        const=False, default=True, action='store_const',
                        help='copy model file to eval directory')
    parser.add_argument('--store-intermediate', dest='storeinterm',
                        const=False, default=True, action='store_const',
                        help='copy model file to eval directory')
    parser.add_argument('--testing', dest='testing',
                        const=True, default=False, action='store_const',
                        help='use only 2000 samples from EMNIST to test')
    parser.add_argument('--use-labels', dest='uselabels',
                        const=True, default=False, action='store_const',
                        help='use labels for training the model (classifier)')

    #parser.add_argument('')

    args = parser.parse_args()
    if not args.model:
        args.model = basename(args.file).split('.')[0]

    # import model from file
    site.addsitedir(dirname(args.file))
    #Model = None
    try:
        print("from " + basename(args.file).split('.')[0] + " import " + args.model + " as NewModel")
        exec("from " + basename(args.file).split('.')[0] + " import " + args.model + " as NewModel")
        Model = locals()['NewModel']
    except:
        parser.print_help()
        exit(-1)

    try:
        mkdir(args.directory)
    except FileExistsError:
        pass

    if args.copy:
        copyfile(args.file, join(args.directory, basename(args.file)))

    torch.random.manual_seed(42)
    try:
        torch.cuda.init()
        device = torch.cuda.current_device()
    except:
        device = torch.device("cpu")

    model = Model().to(device)

    ### Define likelihood averaging
    log_Z = 0.
    def average_loglikelihood(data, func, log_Z=0.):
        res = 0.
        n = 0.
        if args.ais:
            for dat in data:
                res = res + torch.sum(func(dat.to(device), log_Z=log_Z)).detach()
                n += dat.shape[0]
        else:
            for dat in data:
                res = res + torch.sum(func(dat.to(device))).detach()
                n += dat.shape[0]
        return float(res)/n

    ### Load EMNIST dataset

    from emnist import extract_training_samples
    numbers, n_labels = extract_training_samples('digits')
    characters, c_labels = extract_training_samples('letters')

    numbers = torch.tensor(numbers/255., dtype=torch.float).cpu().clone().reshape(-1, 1, 28, 28)
    characters = torch.tensor(characters/255., dtype=torch.float).cpu().clone().reshape(-1, 1, 28, 28)

    if args.testing:
        numbers = numbers[:2000]
        characters = characters[:2000]
    ### Binarization of the data

    if args.binarize:
        numbers[numbers >= 0.5] = 1.
        numbers[numbers < 0.5] = 0.


    ### Split into training, validation and test set

    batch_size_train = 10
    batch_size_valid = 10
    batch_size_test = 10
    train_size = int(3/4*numbers.shape[0])
    valid_size = int(1/8*numbers.shape[0])
    test_size = int(1/8*numbers.shape[0])

    train_data = numbers[:train_size].reshape(train_size//batch_size_train, batch_size_train, 1, 28, 28)
    train_labels = torch.tensor(n_labels[:train_size].reshape(train_size//batch_size_train, batch_size_train))
    valid_data = numbers[train_size:train_size+valid_size].reshape(valid_size//batch_size_valid, batch_size_valid, 1, 28, 28)
    test_data = numbers[train_size+valid_size:].reshape(test_size//batch_size_valid, batch_size_valid, 1, 28, 28)

    # train the model if needed

    trained = True
    try:
        state_dict = torch.load(args.directory + "/model.pt")
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        trained = False
    if not trained or args.retrain:
        valid_loglikelihoods = []
        valid_loglikelihood = -np.inf
        last_valid_loglikelihood = -np.inf
        epoch = 0
        while valid_loglikelihood >= last_valid_loglikelihood or \
              epoch < args.minepochs:
            last_valid_loglikelihood = valid_loglikelihood
            if epoch >= args.maxepochs:
                break
            # Train
            if args.uselabels:
                model.train(data=args.tqdm(train_data),
                            labels=train_labels,
                            epochs=args.valepochs,
                            device=device)
            else:
                model.train(data=args.tqdm(train_data),
                            epochs=args.valepochs,
                            device=device)

            epoch += args.valepochs
            # Evaluate likelihood
            if args.ais:
                log_Z = model.get_log_Z(1000, 1000)
            valid_loglikelihood = average_loglikelihood(args.tqdm(valid_data), model.loglikelihood, log_Z=log_Z)
            valid_loglikelihoods.append(valid_loglikelihood)

            if args.storeinterm:
                torch.save(model.state_dict(), args.directory + "/model_intermediate_{}.pt".format(epoch))

    torch.save(model.state_dict(), args.directory + "/model.pt")

    try:
        with open(directory + "/results.json", "r") as fp:
            resultdict = json.load(fp)
    except:

        if args.ais:
            log_Z = model.get_log_Z(1000, 1000)

        test_loglikelihood = average_loglikelihood(args.tqdm(test_data), model.loglikelihood, log_Z=log_Z)
        train_loglikelihood = average_loglikelihood(args.tqdm(train_data), model.loglikelihood, log_Z=log_Z)
        valid_loglikelihood = average_loglikelihood(args.tqdm(valid_data), model.loglikelihood, log_Z=log_Z)

        resultdict = {"valid_loglikelihood": valid_loglikelihood,
                      "test_loglikelihood": test_loglikelihood,
                      "train_loglikelihood": train_loglikelihood}

        with open(args.directory + "/results.json", "w") as fp:
            json.dump(resultdict, fp)

    samples = model.generate(N=32)
    fig = plt.figure(figsize=[16,8])
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(samples[i].detach().cpu().numpy().reshape(28, 28), cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    fig.savefig(args.directory + "/generated.png")

    fig = plt.figure(figsize=[16,8])

    TP_R, FP_R = build_roc_graph(model, numbers[-1000:].reshape(-1,10,1,28,28).to(device), characters[:1000].reshape(-1,10,1,28,28).to(device), ais = args.ais)

    #print(TP_R, FP_R)

    plot_roc_graph(fig, TP_R, FP_R, directory=args.directory)

    exit(0)

if __name__ == '__main__':
    main()
