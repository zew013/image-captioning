################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy
from nltk.tokenize import word_tokenize
import caption_utils
from typing import Tuple, List

from torch.nn.utils.rnn import pack_padded_sequence

ROOT_STATS_DIR = './experiment_data'
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        # zw
        if config_data['dataset']['transform']:
                self.__train_loader2 = get_datasets(config_data, get_transformed = True)[3]
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)
        self.__transform = config_data['dataset']['transform']
        print(self.__transform)
        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__early_stop = config_data['experiment']['early_stop']
        self.__patience = config_data['experiment']['patience']
        self.__batch_size = config_data['dataset']['batch_size']

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__best_model = deepcopy(self.__model.state_dict())

        # criterion
        self.__criterion = torch.nn.CrossEntropyLoss()

        # optimizer
        self.__optimizer = getattr(torch.optim, config_data['experiment']['optimizer'])\
            (self.__model.parameters(),lr=config_data['experiment']['learning_rate'], weight_decay=0.0001)

        # LR Scheduler
        self.__lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.__optimizer, T_max=self.__epochs)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        # check if training_losses.txt val_losses.txt and latest_model.pt all exist, otherwise delete the directory
        
        training_losses_name = "training_losses.txt"
        val_losses_name = "val_losses.txt"
        latest_model_name = "latest_model.pt"
        
        if os.path.exists(self.__experiment_dir) \
            and os.path.exists(os.path.join(self.__experiment_dir, training_losses_name))\
            and os.path.exists(os.path.join(self.__experiment_dir, val_losses_name))\
            and os.path.exists(os.path.join(self.__experiment_dir, latest_model_name)):
            
            self.__training_losses = read_file_in_dir(self.__experiment_dir, training_losses_name)
            self.__val_losses = read_file_in_dir(self.__experiment_dir, val_losses_name)
            self.__val_losses = read_file_in_dir(self.__experiment_dir, val_losses_name)
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, latest_model_name))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir, exist_ok=True)


    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def val(self):
        return self.__val()
    
    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        patience_count = 0
        min_loss = 100
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')
            print('--------')
            start_time = datetime.now()
            self.__current_epoch = epoch
            print('Training...')
            print('-----------')
            train_loss = self.__train(epoch)
            print('Validating...')
            print('-------------')
            val_loss = self.__val()

            # save best model
            if val_loss < min_loss:
                min_loss = val_loss
                self.__best_model = deepcopy(self.__model.state_dict())

            # early stop if model starts overfitting
            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                if patience_count >= self.__patience:
                    print('\nEarly stopping!')
                    break

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if self.__lr_scheduler is not None:
                self.__lr_scheduler.step()
        self.__model.load_state_dict(self.__best_model)

    def __compute_loss(self, images: torch.Tensor, captions: torch.Tensor, lengths: list, teacher_forcing: bool = True) -> torch.Tensor:
        """
        Computes the loss after a forward pass through the model
        """
        outputs = self.__model(images, captions, lengths, teacher_forcing=teacher_forcing)
        packed_caption = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        # print("outputs shape: ", outputs.shape)
        # print("captions shape: ", captions.shape)
        return self._Experiment__criterion(outputs, packed_caption)

    def __train(self, epoch):
        """
        Trains the model for one epoch using teacher forcing and minibatch stochastic gradient descent
        """
        self.__model.train()
        total_loss = 0
        # zw
        if epoch % 2 == 0 and self.__transform :
            trainloader = self.__train_loader2
            print('transforming')
        else:
            trainloader = self.__train_loader
            print('not transforming')
        with tqdm(total=len(trainloader)) as pbar:
            for i, (images, targets, lengths, img_ids) in enumerate(self.__train_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()

                self.__optimizer.zero_grad()
                loss = self.__compute_loss(images, targets, lengths)
                loss.backward()
                self.__optimizer.step()
                
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
                total_loss += loss.item()

        return total_loss / len(self.__train_loader)

    def __generate_captions(self, img_id, outputs, testing) -> Tuple[List[str], List[str]]:
        """
        Generate captions without teacher forcing
        Params:
            img_id: Image Id for which caption is being generated
            outputs: output from the forward pass for this img_id
            testing: whether the image_id comes from the validation or test set
        Returns:
            tuple (list of original captions, predicted caption)
        """
        coco = self.__coco_test if testing else self.__coco

        img_captions = coco.imgToAnns[img_id]
        original_captions = [caption['caption'] for caption in img_captions]
        predicted_caption = self.__vocab.decode(outputs)
        
        # join the predicted caption with spaces (between <start> and <end> tokens)
        
        start_idx, end_idx = 0, -1
        try:
            start_idx, end_idx = predicted_caption.index('<start>'), predicted_caption.index('<end>')
        except ValueError:
            pass
            
        
        predicted_caption = " ".join(predicted_caption[start_idx + 1:end_idx])
        
        
        return original_captions, predicted_caption

    def __str_captions(self, img_id, original_captions, predicted_caption):
        """
            !OPTIONAL UTILITY FUNCTION!
            Create a string for logging ground truth and predicted captions for given img_id
        """
        result_str = "Captions: Img ID: {},\nActual: {},\nPredicted: {}\n".format(
            img_id, original_captions, predicted_caption)
        return result_str

    def __val(self):
        """
        Validate the model for one epoch using teacher forcing
        """
        self.__model.eval()
        total_loss = 0
        with torch.no_grad():
            with tqdm(total=len(self.__val_loader)) as pbar:
                for i, (images, targets, lengths, img_ids) in enumerate(self.__val_loader):

                    
                    if torch.cuda.is_available():
                        images = images.cuda()
                        targets = targets.cuda()

                    loss = self.__compute_loss(images, targets, lengths, teacher_forcing=True) # still using teacher forcing
                    total_loss += loss.item()
                    
                    
                    # for the first batch, generate captions and log them
                    if i == 0:
                        res = self.__model(images, targets, lengths, teacher_forcing=False)
                        print(res) 
                        for j in range(0, len(res)):
                            original_captions, predicted_caption = self.__generate_captions(img_ids[j], res[j], testing=False)
                            result_str = self.__str_captions(img_ids[j], original_captions, predicted_caption)
                            print(result_str)
                    
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)



        return total_loss / len(self.__val_loader)

    def test(self):
        """
        Test the best model on test data. Generate captions and calculate bleu scores
        """
        # self.__model.load_state_dict(self.__best_model)
        self.__model.eval()
        total_loss = 0
        blue1_scores = []
        blue4_scores = []
        with torch.no_grad():
            with tqdm(total=len(self.__test_loader)) as pbar:
                for i, (images, targets, lengths, img_ids) in enumerate(self.__test_loader):
                    if torch.cuda.is_available():
                        images = images.cuda()
                        targets = targets.cuda()
                        
                    loss = self.__compute_loss(images, targets, lengths, teacher_forcing=True) # still using teacher forcing
                    total_loss += loss.item()
                        
                    res = self.__model(images, targets, lengths, teacher_forcing=False)
                    for j in range(0, len(res)):
                        original_captions, predicted_caption = self.__generate_captions(img_ids[j], res[j], testing=True)
                        tokenized_original_captions = caption_utils.tokenize(original_captions)
                        tokenized_predicted_caption = caption_utils.tokenize(predicted_caption)
                        
                        blue1_score = caption_utils.bleu1(tokenized_original_captions, tokenized_predicted_caption)
                        blue4_score = caption_utils.bleu4(tokenized_original_captions, tokenized_predicted_caption)
                        
                        # result_str = self.__str_captions(img_ids[j], original_captions, predicted_caption)
                        # print("BLEU1: ", blue1_score)
                        # print("BLEU4: ", blue4_score)
                    
                        blue1_scores.append(blue1_score)
                        blue4_scores.append(blue4_score)
                    pbar.update(1)
                    pbar.set_postfix(blue1=np.mean(blue1_scores), blue4=np.mean(blue4_scores))
                        
                    # captions = [[] for _ in range(images.size(0))] # list of list of predicted captions and blue1 score pair
                    
                     
                    # for k in range(10): # generate 10 captions for each image
                    #     res = self.__model(images, targets, lengths, teacher_forcing=False)
                    #     for j in range(0, len(res)):
                    #         original_captions, predicted_caption = self.__generate_captions(img_ids[j], res[j], testing=True)
                    #         if k == 0:
                    #             print(f"Original ({j}): ", original_captions)
                    #         tokenized_original_captions = caption_utils.tokenize(original_captions)
                    #         tokenized_predicted_caption = caption_utils.tokenize(predicted_caption)
                            
                    #         blue1_score = caption_utils.bleu1(tokenized_original_captions, tokenized_predicted_caption)
                    #         blue4_score = caption_utils.bleu4(tokenized_original_captions, tokenized_predicted_caption)
                            
                    #         # result_str = self.__str_captions(img_ids[j], original_captions, predicted_caption)
                    #         # print("BLEU1: ", blue1_score)
                    #         # print("BLEU4: ", blue4_score)
                        
                    #         avg_bleu1 += blue1_score
                    #         avg_bleu4 += blue4_score
                            
                    #         captions[j].append((predicted_caption, blue1_score))
                    
                    # captions = [sorted(captions[j], key=lambda x: x[1], reverse=True) for j in range(len(captions))]

                    # # print(*((i, cs) for i, cs in enumerate(captions)), sep='\n')
                    
                    # break
                        
                    # avg_bleu1 /= len(res)
                    # avg_bleu4 /= len(res)
                    
                    # print("Avg BLEU1: ", avg_bleu1)
                    # print("Avg BLEU4: ", avg_bleu4)
                    
                    # pbar.set_postfix(bleu1=avg_bleu1, bleu4=avg_bleu4)
                    # pbar.update(1)
                    
        print("Avg BLEU1: ", sum(blue1_scores) / len(blue1_scores))
        print("Avg BLEU4: ", sum(blue4_scores) / len(blue4_scores))
        
        # create a histogram for the bleu scores
        
        # plt.hist(blue1_scores, bins=20)
        # plt.title("BLEU1 Scores")
        # # save the figure
        # plt.savefig("bleu1_scores.png")
        
        # plt.hist(blue4_scores, bins=20)
        # plt.title("BLEU4 Scores")
        # plt.savefig("bleu4_scores.png")

        print('Test loss: {}'.format(total_loss / len(self.__test_loader)))

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
