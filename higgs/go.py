from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
import logging

from utils import TextLoader as TextLoader_HIGGS
from model import Model
# from model_semi import Model as Model_semi
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_curve, auc

def mylog(msg):
  print(msg)
  logging.info(msg)
  return


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='../../../data/gender_combine',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--task', type=str, default='discover',
                        help='discover or HIGGS')
    parser.add_argument('--source', type=str, default=1,
                        help='0: username; 1: engagement; 2: first name. 0,1,2; 0,2')
    parser.add_argument('--timestamp', type=str, default='1026',
                        help='dataset created')
    parser.add_argument('--lstm_input', type=str, default=0,
                        help='0, or 1 or 0,1')    
    parser.add_argument('--input_dim', type=int, default=0,
                        help='dimension of source 0')
    parser.add_argument('--input_dim2', type=int, default=0,
                        help='dimension of source 1')
    parser.add_argument('--input_dim3', type=int, default=0,
                        help='dimension of source 2')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='margin used in mul+')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='l2 weight decay')
    parser.add_argument('--mul', type=int, default=0,
                        help='0: simplified, 1: geo_mean, 2: min')

    parser.add_argument('--late_attention', type=int, default=0,
                        help='0 to disable attention, i.e., average')
    parser.add_argument('--dnn_first', type=int, default=0,
                        help='for engagement component')
    parser.add_argument('--early_attention_rnn', type=int, default=0,
                        help='0 to disable attention, i.e., average')
    parser.add_argument('--sep_attend_early', type=int, default=1,
                        help='1: separate attention module for different modules')
    parser.add_argument('--sep_attend', type=int, default=1,
                        help='1: separate attention module for different modules')
    parser.add_argument('--output_size', type=int, default=256,
                        help='size of output')
    parser.add_argument('--hidden_nl', type=int, default=1,
                        help='1: nl in hidden')
    parser.add_argument('--dff', type=int, default=0,
                        help='1: adding a regularizer to make component vectors different')
    parser.add_argument('--dffeta', type=float, default=0.0,
                        help='eta: balancing regularizer')
    parser.add_argument('--size', type=int, default=128,
                        help='size of embedding state')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--dnn_size', type=str, default='500,200',
                        help='size of DNN hidden state')
    parser.add_argument('--dnn_size2', type=str, default='300,300',
                        help='size of DNN2 hidden state')
    parser.add_argument('--layerwise', type=str, default='',
                        help='layers to choose from')    
    parser.add_argument('--ngram', type=int, default=1,
                        help='Ngram as input')
    parser.add_argument('--ngram2', type=int, default=0,
                        help='Ngram2 as input')    
    parser.add_argument('--attention_size', type=int, default=0,
                        help='0 to disable attention, i.e., average')
    parser.add_argument('--collaborative', type=int, default=0,
                        help='0 to enable collaborative training')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--num_fc_layers', type=int, default=2,
                        help='number of output layers')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--act', type=str, default='relu',
                        help='relu, tanh')
    parser.add_argument('--test', type=str, default='',
                        help='test or null')    
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='adam, adagrad, sgd')    

    parser.add_argument('--birnn', type=int, default=0,
                        help='bidirectional or not')
    parser.add_argument('--cross_attend', type=int, default=0,
                        help='attention weights look at all')    
    parser.add_argument('--semi', type=int, default=0,
                        help='semi')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='balance')
    parser.add_argument('--batchnorm', type=int, default=0,
                        help='batch normalization or not')    
    parser.add_argument('--no_male_prob', type=int, default=0,
                        help='1: no male prob feature')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=15,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--overlap', type=int, default=1,
                        help='overlapping of artificial modalities')
    parser.add_argument('--eval_every', type=int, default=5000,
                        help='save frequency')
    parser.add_argument('--print_every', type=int, default=500,
                        help='print frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stop')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--mode', type=str, default='train', 
                        help='train or decode')
    args = parser.parse_args()

    _name = (str(time.localtime().tm_mon) + str(time.localtime().tm_mday) 
        + '-' + str(args.timestamp)
        + 'feat'+ str(args.source) 
        + args.model 
        + 'l' + str(args.num_layers) + 'h' + str(args.size) 
        + str(args.optimizer)
        + 'mb' + str(args.batch_size) +'lr' + str(args.learning_rate) + 'd'
        + str(args.keep_prob)
        + 'gc' + str(args.grad_clip) + 'dc' + str(args.decay_rate) +'a' + 
        str(args.attention_size) 
        + 'wd' + str(args.wd)
        + 'late' + str(args.late_attention) + 'sep' + str(args.sep_attend) 
        + 'eatt' + str(args.early_attention_rnn) +'esep' + str(args.sep_attend_early)
        + 'ho' + str(args.output_size) + 'col' + str(args.collaborative)
        + 'mul' + str(args.mul) + 'margin' + str(args.margin)
        + 'bn' + str(args.batchnorm) +'df' + str(args.dnn_first) +'hiddennl' + str(args.hidden_nl)
        + 'ca' + str(args.cross_attend) +'dff' + str(args.dff) + 'eta' + str(args.dffeta)
        + 'overlap' + str(args.overlap) + args.act +'lw' + args.layerwise
        + args.test)

    if args.task == 'HIGGS':
        _name = 'HIGGS' + _name

    layer1 = args.dnn_size.split(',')
    if len(layer1) == 1 and int(layer1[0]) == 0:
        args.num_layers = 0
    else:
        args.num_layers = len(layer1)

    # if args.num_fc_layers > 0:
    _name += 'fc' + str(args.num_fc_layers) + '-' + str(args.dnn_size)
    _name += 'fc2' + str(args.dnn_size2)
    # if args.no_male_prob == 1:
    #     _name += 'nofn'
    args.save_dir = os.path.join(args.save_dir, _name)
    args.log_dir  = os.path.join(args.log_dir, _name)


    # if args.test is not '':
    #     args.print_every = 50
    #     args.eval_every = 500
    assert(args.mode in ['train', 'decode'])
    if args.mode == 'train':
        train(args)
    elif args.mode == 'decode':
        exit(0)


def train(args):
    if args.task == 'HIGGS':
        print('task: HIGGS')
        TextLoader = TextLoader_HIGGS
    else:
        print('task: discover')

    sources = args.source.split(',')
    assert(len(sources) > 0)
    sources = [int(s) for s in sources]

    data_loader = TextLoader(args.data_dir, args.batch_size, 
        test=args.test, timestamp = args.timestamp, overlap=args.overlap)

    if len(sources) == 3:
        args.input_dim, args.input_dim2, args.input_dim3 = (
            data_loader.x_tr.shape[1], 
            data_loader.z_tr.shape[1], 
            data_loader.v_tr.shape[1])
    else:
        if 0 in sources:
            args.input_dim = data_loader.x_tr.shape[1]
        if 1 in sources:
            args.input_dim2= data_loader.z_tr.shape[1]
        if 2 in sources:
            args.input_dim3= data_loader.v_tr.shape[1]

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    ##
    # with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
    #     cPickle.dump((data_loader.chars, data_loader.vocab), f)

    log_path = os.path.join(args.save_dir,"log.txt")
    logging.basicConfig(filename=log_path,level=logging.DEBUG)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None):
            model = Model(args, is_training=True)
    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            mval = Model(args, is_training=False, keep_prob=model.keep_prob,
                inp=[model.input_data, model.input_data2, model.input_data3],
                labels=model.labels)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        step_time, train_loss_accu, acc_accum, f1_accum = 0.0, 0.0, 0.0, 0.0
        xent_accu = 0.0
        auc_accum = 0.0
        pred_accum = []
        label_accum = []

        best_acc, final_f1= 0.0, 0.0
        best_obj = 1000000000
        best_auc = 0.0

        patience = args.patience
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            step_time, train_loss_accu, acc_accum, f1_accum = 0.0, 0.0, 0.0, 0.0
            xent_accu = 0.0
            auc_accum = 0.0

            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            n_batches = len(data_loader.y_tr) / data_loader.batch_size

            args.eval_every = int(0.25 * n_batches)
            args.print_every = int(0.02 * n_batches)

            for b in range(n_batches):
                start = time.time()
                if args.semi:
                    x, y, z, v, ts, ws = data_loader.next_batch(
                        split='train', append=args.pad, semi=True)
                else:
                    x, y, z, v = data_loader.next_batch(split='train')

                feed = {model.labels: y}
                if 0 in sources:
                    feed[model.input_data] = x
                if 1 in sources:
                    feed[model.input_data2] = z
                if 2 in sources:
                    feed[model.input_data3] = v
                feed[model.keep_prob] = args.keep_prob
                # feed[model.training] = True
                if args.semi:
                    feed[model.targets] = ts
                    feed[model.weights] = ws

                # instrument for tensorboard
                if len(sources) == 1:
                    res = sess.run([summaries, model.cost, model.pred, model.xent] + model.train_and_update_op, feed)
                    summ, train_loss, pred, xent = res[0], res[1], res[2], res[3]
                else:
                    if args.semi:
                        # outdated
                        summ, train_loss, pred, _, l1, l2 = sess.run([summaries, model.cost, model.pred, model.train_op, model.loss_recon, model.loss_pred], feed)
                    elif args.collaborative == 0:
                        
                        res = sess.run(
                            [summaries, model.cost, model.loss_v, model.pred, model.xent, model.train_op_v]+model.train_and_update_op, feed)
                        summ, train_loss, v_l,  pred, xent = res[0], res[1], res[2], res[3], res[4]
                    else:
                        res = sess.run([summaries, model.cost, model.loss_v, model.pred, model.xent ]+model.train_and_update_op, feed)
                        summ, train_loss, v_l,  pred, xent = res[0], res[1], res[2], res[3], res[4]

                writer.add_summary(summ, e * n_batches + b)
                y_true = [k[0] for k in y]
                batch_acc = accuracy_score(y_true, 1-pred)
                batch_f1 = f1_score(y_true, 1-pred)
                # fpr, tpr, thresh = roc_curve(y_true, 1-pred, pos_label=1)
                # batch_auc = auc(fpr, tpr)

                # print(type(pred))
                # print(pred.shape)
                pred_accum.extend(list(1-pred))
                label_accum.extend(y_true)

                end = time.time()
                step_time += end - start
                train_loss_accu += train_loss
                xent_accu += xent
                acc_accum += batch_acc
                f1_accum += batch_f1
                # auc_accum += batch_auc
                if args.semi:
                    l1_accu += l1
                    l2_accu += l2

                if b % args.print_every == 0 and b > 0:
                    average_step_time = step_time / args.print_every
                    train_loss_accu /= args.print_every
                    xent_accu /= args.print_every
                    acc_accum /= args.print_every
                    f1_accum /= args.print_every
                    # auc_accum /= args.print_every
                    # fpr, tpr, thresh = roc_curve(label_accum, pred_accum, pos_label=1)
                    # auc_accum = auc(fpr, tpr)
                    auc_accum = 0

                    if args.semi:
                        l1_accu /= args.print_every
                        l2_accu /= args.print_every
                    if args.semi:
                        mylog("{}/{} (epoch {}), train_loss = {:.3f}, acc = {:.3f}, f1 = {:.3f}, auc = {:.3f}, time/batch = {:.3f}, {:.3f}-{:.3f}"
                              .format(e * n_batches + b,
                                      args.num_epochs * n_batches,
                                      e, train_loss_accu, acc_accum, f1_accum, auc_accum, average_step_time, l1_accu, l2_accu))
                    else:
                        mylog("{}/{} (epoch {}), train_loss = {:.3f} (xent={:.3f}), acc = {:.3f}, f1 = {:.3f}, auc = {:.3f}, time/batch = {:.3f}"
                              .format(e * n_batches + b,
                                      args.num_epochs * n_batches,
                                      e, train_loss_accu, xent_accu, acc_accum, f1_accum, auc_accum, average_step_time))
                    step_time, train_loss_accu, acc_accum, f1_accum = 0.0, 0.0, 0.0, 0.0
                    auc_accum = 0.0
                    xent_accu = 0.0
                    label_accum , pred_accum = [], []

                    if args.semi:
                        l1_accu, l2_accu = 0.0, 0.0


                # test
                if (e * n_batches + b) % args.eval_every == 0 and (e * n_batches + b) > 0:
                    # eval on test set
                    if args.semi:
                        test_loss, test_acc, test_f1, test_auc, y_true, y_pred, probs, l1, l2 = model_eval(sess, mval, data_loader, args, semi=True)
                    else:
                        test_loss, test_acc, test_f1, test_auc, y_true, y_pred, probs, alphas = model_eval(sess, mval, data_loader, args)

                    if args.semi:
                        mylog("{}/{} (epoch {}), test_loss = {:.3f}, test_acc = {:.3f}, test_f1 = {:.3f}, test_auc = {:.3f}, losses: {:.3f}, {:.3f}"
                              .format(e * n_batches + b,
                                      args.num_epochs * n_batches,
                                      e, test_loss, test_acc, test_f1, test_auc, l1, l2))
                    else:
                        mylog("{}/{} (epoch {}), test_loss = {:.3f}, test_acc = {:.3f}, test_f1 = {:.3f}, test_auc = {:.3f}"
                              .format(e * n_batches + b,
                                      args.num_epochs * n_batches,
                                      e, test_loss, test_acc, test_f1, test_auc))
                    
                    if best_obj > test_loss:
                        best_obj = test_loss
                        best_acc = test_acc
                        final_f1 = test_f1
                        best_auc = test_auc
                        final_probs = probs
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, 
                            global_step=e * n_batches + b)
                        mylog("better model saved to {}".format(checkpoint_path))
                        with open(os.path.join(args.save_dir, 'test_prediction.pkl'), 'wb') as f:
                            cPickle.dump((y_true, y_pred, probs), f)
                        if not args.semi:
                            with open(os.path.join(args.save_dir, 'test_alphas.pkl'), 'wb') as f:
                                cPickle.dump((alphas), f)
                        filename = os.path.join(args.save_dir, 'probs.result.test')
                        with open(filename, 'wb') as f:
                            cPickle.dump(final_probs, f)
                        patience = args.patience
                    else:
                        patience -=1

                    mylog("current best: obj: {:.3f}, acc: {:.3f}, f1: {:.3f}, auc: {:.3f}"
                          .format(best_obj, best_acc, final_f1, best_auc))

                    if patience <= 0 :
                        mylog("patience exhausted. \nfinal results: {:.3f}, {:.3f}".format(best_acc, final_f1, best_auc))
                        if args.semi:
                            res_train = model_eval(sess, mval, data_loader, args, 'train', semi=True)
                        else:
                            res_train = model_eval(sess, mval, data_loader, args, 'train', semi=False)
                        train_loss = res_train[0]
                        train_acc = res_train[1]
                        train_f1 = res_train[2]
                        train_auc = res_train[3]

                        mylog("final results on train: {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(train_loss, train_acc, train_f1, train_auc))
                        probs = res_train[6]    
                        filename = os.path.join(args.save_dir, 'probs.result.train')
                        with open(filename, 'wb') as f:
                            cPickle.dump(probs, f)
                        exit(0)

def model_eval(sess, model, data_loader, args, data='test', semi=False):

    sources = args.source.split(',')
    assert(len(sources) > 0)
    sources = [int(s) for s in sources]

    test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0
    test_auc = 0.0
    if semi:
        l1_accu, l2_accu = 0.0, 0.0
    if data == 'train':
        n_batches_te = len(data_loader.y_tr) / data_loader.batch_size
    else:
        n_batches_te = len(data_loader.y_te) / data_loader.batch_size
    data_loader.reset_batch_pointer()
    labels, predictions, probs = [], [], []
    all_alphas = []
    for b_te in range(n_batches_te):
        if semi:
            x, y, z, v, ts, ws = data_loader.next_batch(eval=True, split=data, semi=True)
        else:
            x, y, z, v = data_loader.next_batch(eval=True, split=data)
        feed = {model.labels: y}
        if 0 in sources:
            feed[model.input_data] = x
        if 1 in sources:
            feed[model.input_data2] = z
        if 2 in sources:
            feed[model.input_data3] = v
        feed[model.keep_prob]= 1.0
        # feed[model.training] = False
        if semi:
            feed[model.targets] = ts
            feed[model.weights] = ws
        if len(sources) == 1:
            batch_loss, batch_pred, batch_probs= sess.run([model.cost, model.pred, model.probs], feed)            
            attri = []
        else:
            if semi:
                batch_loss, batch_pred, batch_probs, l1, l2= sess.run([model.cost, model.pred, model.probs, model.loss_recon, model.loss_pred], feed)
            else:
                if args.collaborative != 0:
                    batch_loss, batch_pred, batch_probs, attri= sess.run([model.cost, model.pred, model.probs, model.ps], feed)
                else:
                    batch_loss, batch_pred, batch_probs, attri= sess.run([model.cost, model.pred, model.probs, model.loss_v], feed)
                all_alphas.append(attri)
        

        batch_probs = [p[0] for p in batch_probs]
        y_true = [k[0] for k in y]
        batch_acc = accuracy_score(y_true, 1- batch_pred)
        batch_f1 = f1_score(y_true, 1 - batch_pred)
        fpr, tpr, thresh = roc_curve(y_true, 1-batch_pred, pos_label=1)
        batch_auc = auc(fpr, tpr)

        test_loss += batch_loss
        test_acc += batch_acc
        test_f1 += batch_f1
        # test_auc += batch_auc
        if semi:
            l1_accu += l1
            l2_accu += l2

        labels.extend(y_true)
        predictions.extend(batch_pred)
        probs.extend(batch_probs)

    test_loss /= n_batches_te    
    test_f1 /= n_batches_te
    test_acc /= n_batches_te 

    # test_auc /= n_batches_te   

    fpr, tpr, thresh = roc_curve(labels, probs, pos_label=1)
    test_auc = auc(fpr, tpr)

    if semi:
        l1_accu /= n_batches_te
        l2_accu /= n_batches_te
        return test_loss, test_acc, test_f1, test_auc, labels, predictions, probs, l1_accu, l2_accu
    else:        
        return test_loss, test_acc, test_f1, test_auc, labels, predictions, probs, zip(*all_alphas)


if __name__ == '__main__':
    main()
