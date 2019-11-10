import logging
import random

import ast
import numpy as np
from scipy import integrate
import torch
import fire

from utils.argparse import ConfigurationParer
from inputs.statistic import dataset_statistic
from inputs.sequence_generator import load_sequences, generate_sequence
from inputs.batch_iterator import PaddedBatchIterator
import models

logger = logging.getLogger(__name__)


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    parser.add_run_cfgs()

    cfg = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(cfg.seed)

    # mode
    if cfg.mode == 0:
        statistic(cfg)
    elif cfg.mode == 1:
        preprocessing(cfg)
    elif cfg.mode == 2:
        training(cfg)
    elif cfg.mode == 3:
        testing(cfg)


def statistic(cfg):
    print('Dataset statistic starting...')
    domain_dict = ast.literal_eval(cfg.domain_dict)
    dataset_statistic(cfg.csv_file, domain_dict, cfg.dataset_name, cfg.dataset_dir)
    print('Dataset statistic finished.')


def preprocessing(cfg):
    print('Preprocessing starting...')
    domain_dict = ast.literal_eval(cfg.domain_dict)
    # generate_time_sequence(cfg.CSV_FILE, domain_dict, cfg.TIME_FILE,
    #                        cfg.TRAIN_TIME_FILE, cfg.DEV_TIME_FILE,
    #                        train_rate=cfg.TRAIN_RATE, min_length=cfg.MIN_LENGTH, max_length=cfg.MAX_LENGTH)
    # generate_event_sequence(cfg.CSV_FILE, domain_dict, cfg.EVENT_FILE, cfg.TRAIN_EVENT_FILE, cfg.DEV_EVENT_FILE,
    #                         cfg.EVENT_INDEX_FILE, train_rate=cfg.TRAIN_RATE,
    #                         min_length=cfg.MIN_LENGTH, max_length=cfg.MAX_LENGTH)
    generate_sequence(cfg.csv_file,
                      domain_dict,
                      cfg.time_file,
                      cfg.train_time_file,
                      cfg.dev_time_file,
                      cfg.event_file,
                      cfg.train_event_file,
                      cfg.dev_event_file,
                      cfg.event_index_file,
                      train_rate=cfg.train_file,
                      min_length=cfg.min_length,
                      max_length=cfg.max_length,
                      min_event_interval=cfg.min_event_interval)
    print('Preprocessing finished.')


def training(cfg):
    print('Training starting...')

    # pytorch seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device > -1 and not torch.cuda.is_available():
        logger.error('config conflicts: no gpu available, use cpu for training.')
        cfg.device = -1
    if cfg.device > -1:
        torch.cuda.manual_seed(cfg.seed)

    # load dataset
    train_sequences = load_sequences(cfg.train_time_file, cfg.train_event_file)
    train_batch_iterator = PaddedBatchIterator(train_sequences, cfg.mark, cfg.diff,
                                               cfg.save_last_time)
    dev_sequences = load_sequences(cfg.dev_time_file, cfg.dev_event_file)
    dev_batch_iterator = PaddedBatchIterator(dev_sequences, cfg.mark, cfg.diff, cfg.save_last_time)

    # model
    model = getattr(models, cfg.model)(cfg)
    if cfg.continue_training:
        checkpoint = torch.load(cfg.last_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = model.cuda(device=cfg.device)

    # citeration
    citeration = getattr(models, cfg.model + 'Loss')(cfg)

    # optimizer
    # model_optimizer = torch.optim.Adam(model.parameters(),
    #                                    lr=cfg.learning_rate,
    #                                    betas=(cfg.adam_beta1, cfg.adam_beta2),
    #                                    weight_decay=cfg.weight_decay)
    # citeration_optimizer = torch.optim.Adam(citeration.parameters(),
    #                                         lr=cfg.learning_rate,
    #                                         betas=(cfg.adam_beta1, cfg.adam_beta2),
    #                                         weight_decay=cfg.weight_decay)

    model_optimizer = torch.optim.RMSprop(model.parameters(),
                                          lr=cfg.learning_rate,
                                          eps=cfg.adam_epsilon)
    model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=30, gamma=0.1)

    if cfg.save_last_time:
        citeration_optimizer = torch.optim.RMSprop(citeration.parameters(),
                                                   lr=cfg.learning_rate,
                                                   eps=cfg.adam_epsilon)
        citeration_scheduler = torch.optim.lr_scheduler.StepLR(citeration_optimizer,
                                                               step_size=30,
                                                               gamma=0.1)

    # meters
    loss_meter = []
    max_event_f1 = None
    max_event_precision = None
    max_event_recall = None
    max_event_acc = None
    min_time_loss = None
    epoch_cnt = 0

    for epoch in range(cfg.epoches):
        model.train()
        model_scheduler.step()
        if cfg.save_last_time:
            citeration_scheduler.step()
        train_batch_iterator.shuffle()
        batch_id = 1
        while True:
            end, input, target, last_time, length = train_batch_iterator.next_batch(
                cfg.train_batch_size)
            model_optimizer.zero_grad()
            if cfg.save_last_time:
                citeration_optimizer.zero_grad()
            output = model.forward(input, length)
            batch_loss = citeration(output, target)
            loss_meter.append(batch_loss[2].item())
            batch_loss[2].backward()
            model_optimizer.step()
            if cfg.save_last_time:
                citeration_optimizer.step()
            if batch_id % cfg.validate_every == 0:
                print("epoch: %d\tbatch_id:%d\tloss:%f\ttime loss: %f\tevent loss: %f" %
                      (epoch, batch_id, np.array(loss_meter).mean(), batch_loss[0].item(),
                       batch_loss[1].item()))
                loss_meter = []
            batch_id += 1
            if end: break

        model.eval()
        with torch.no_grad():
            event_total = 0
            all_cnt = np.zeros(cfg.event_classes)
            acc_cnt = np.zeros(cfg.event_classes)
            pre_cnt = np.zeros(cfg.event_classes)
            time_error = 0
            dev_batch_iterator.shuffle()
            while True:
                end, input, target, last_time, length = dev_batch_iterator.next_batch(
                    cfg.test_batch_size)
                output = model.forward(input, length)
                event_total += length.shape[0]

                event_output = output[1].cpu().numpy()
                event_output = np.argmax(event_output, axis=1).astype(int)
                event_target = target[:, 1].astype(int)
                for idx in range(event_target.shape[0]):
                    all_cnt[event_target[idx]] += 1
                    pre_cnt[event_output[idx]] += 1
                    if event_output[idx] == event_target[idx]:
                        acc_cnt[event_output[idx]] += 1

                if last_time is None:
                    time_output = output[0].squeeze().cpu().numpy()
                    time_target = target[:, 0]
                    time_error += np.abs(time_output - time_target).sum()
                else:
                    time_target = target[:, 0]
                    history_event = output[0].squeeze().cpu().numpy()
                    intensity_w = citeration.intensity_w.cpu().data.numpy()
                    intensity_b = citeration.intensity_b.cpu().data.numpy()
                    next_time = np.array([
                        integrate.quad(
                            lambda t: (t + last_time[idx]) * np.
                            exp(history_event[idx] + intensity_w * t + intensity_b +
                                (np.exp(history_event[idx] + intensity_b) - np.exp(history_event[
                                    idx] + intensity_w * t + intensity_b)) / intensity_w), 0,
                            np.inf)[0] for idx in range(history_event.shape[0])
                    ])
                    time_error += np.abs(next_time - last_time - time_target).sum()
                if end:
                    break
            print(acc_cnt, acc_cnt.sum())
            print(pre_cnt, pre_cnt.sum())
            print(all_cnt, all_cnt.sum())
            cnt = 0
            score = 0.0
            for idx in range(cfg.event_classes):
                if all_cnt[idx] != 0:
                    cnt += 1
                    score += acc_cnt[idx] / all_cnt[idx]
            event_recall = score / cnt
            cnt = 0
            score = 0.0
            for idx in range(cfg.event_classes):
                if pre_cnt[idx] != 0:
                    cnt += 1
                    score += acc_cnt[idx] / pre_cnt[idx]
            event_precision = score / cnt
            cnt = 0
            score = 0.0
            for idx in range(cfg.event_classes):
                if all_cnt[idx] != 0 and pre_cnt[idx] != 0:
                    cnt += 1
                    precision = acc_cnt[idx] / pre_cnt[idx]
                    recall = acc_cnt[idx] / all_cnt[idx]
                    score += ((2 * precision * recall) / (precision + recall))
            event_f1 = score / cnt
            event_acc = acc_cnt.sum() / all_cnt.sum()
            print("epoch: %d\tevent_recall: %f\tevent_precision: %f\tevent_f1: %f\ttime_error: %f" %
                  (epoch, event_recall, event_precision, event_f1, time_error / event_total))
            print(
                "-------------------------------------------------------------------------------------"
            )

            if max_event_f1 is None or event_f1 > max_event_f1:
                max_event_f1 = event_f1
                best_model = {
                    'epoch': epoch,
                    'precision': event_precision,
                    'recall': event_recall,
                    'f1': event_f1
                }
                epoch_cnt = 0
                # torch.save({'epoch': epoch,
                #             'model_state_dict': model.state_dict(),
                #             'model_optimizer_state_dict': model_optimizer.state_dict(),
                #             # 'citeration_optimizer_state_dict': citeration_optimizer.state_dict(),
                #             'citeration': citeration.state_dict()}, cfg.BEST_MODEL)
            else:
                epoch_cnt += 1

            if max_event_precision is None or event_precision > max_event_precision:
                max_event_precision = event_precision

            if max_event_recall is None or event_recall > max_event_recall:
                max_event_recall = event_recall

            if min_time_loss is None or time_error / event_total < min_time_loss:
                min_time_loss = time_error / event_total

            if max_event_acc is None or event_acc > max_event_acc:
                max_event_acc = event_acc

            if epoch_cnt > cfg.early_stop:
                break

    print('best model:', best_model)
    print("max_event_precision: %f\tmax_event_recall: %f\tmax_event_acc: %f\tmin_time_loss: %f" %
          (max_event_precision, max_event_recall, max_event_acc, min_time_loss))
    print('training finished.')


def testing(cfg):
    print('testing starting...')
    print('testing finished.')


if __name__ == '__main__':
    main()
