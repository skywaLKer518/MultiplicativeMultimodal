import tensorflow as tf
import numpy as np
from attention import attention, merge, merge2, merge_attend 

from network_var import weight_variable_cpu, batch_norm

import logger

log = logger.get()


class Model():
    def __init__(self, args, keep_prob=None, inp=None, labels=None, is_training=True):
        ''' initialize the model with args '''
        self._bn_update_ops = []
        self.args = args
        self.num_classes = 2
        args.num_filters = args.rnn_size
        self.dtype=tf.float32
        
        # self.training = tf.placeholder(tf.bool, [], name='training')
        self.training = is_training
        self.is_training= self.training

        if inp is None:
            self.keep_prob = tf.placeholder(tf.float32, [])
            ''' input data for three modalities '''
            self.input_data = tf.placeholder(
                tf.float32, [args.batch_size, args.input_dim])
            self.input_data2 = tf.placeholder(
                tf.float32, [args.batch_size, args.input_dim2])
            self.input_data3 = tf.placeholder(
                tf.float32, [args.batch_size, args.input_dim3])

            self.labels = tf.placeholder(
                tf.int32, [args.batch_size, self.num_classes])
        else:
            self.keep_prob = keep_prob
            self.input_data = inp[0]
            if len(inp) > 1:
                self.input_data2 = inp[1]
            if len(inp) > 2:
                self.input_data3 = inp[2]
            self.labels = labels

        sources = args.source.split(',')
        assert(len(sources) > 0)
        sources = [int(s) for s in sources]

        if len(sources) == 1:
            if 0 in sources:
                output = self._layers('single0', self.input_data, 
                    args.dnn_size, keep_prob=self.keep_prob,
                    bn=args.batchnorm,is_training=self.training,
                    act=args.act)

                # output, _ = layers(self.input_data, args.batch_size, 
                #     args.input_dim, args.num_fc_layers, 
                #     args.dnn_size, keep_prob=self.keep_prob, bn=args.batchnorm, training=self.training, act=args.act)
            elif 1 in sources:
                output = self._layers('single1', self.input_data2, 
                    args.dnn_size2, keep_prob=self.keep_prob,
                    bn=args.batchnorm,is_training=self.training,
                    act=args.act)
                # output, _ = layers(self.input_data2, args.batch_size, 
                #     args.input_dim2, args.num_fc_layers, 
                #     args.dnn_size2, keep_prob=self.keep_prob, bn=args.batchnorm, training=self.training, act=args.act)
            # output_size = output.get_shape()[-1].value
            # with tf.variable_scope('output_layer'):
            #     softmax_w = tf.get_variable("softmax_w",
            #                                 [output_size, self.num_classes])
            #     softmax_b = tf.get_variable("softmax_b", [self.num_classes])

            with tf.variable_scope('single_output'):
                self.logits = self._fully_connected(output, self.num_classes)

            # self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            self.pred = tf.argmax(self.logits, 1)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels = self.labels))

        else:
            # combine

            if args.layerwise != '':
                ## void 
                # deprecated
                takeouts = [int(l) for l in args.layerwise.split(',')]
                if len(sources) == 2:
                    input_ = tf.concat([self.input_data, self.input_data2], 1)
                    dim_ = args.input_dim + args.input_dim2
                hiddenlayer, _, output_ = layers1(input_, args.batch_size, dim_, args.num_fc_layers,
                    args.dnn_size, keep_prob=self.keep_prob, bn=args.batchnorm,
                    training=self.training, act=args.act, takeout=takeouts)
                sources = range(len(output_))
                if len(sources) == 2:
                    output1, output2 = output_
                    output3 = None
                elif len(sources) == 3:
                    output1, output2, output3 = output_
            else:
                if 0 in sources:
                    output1 = self._layers('modal0', self.input_data, 
                        args.dnn_size, keep_prob=self.keep_prob,
                        bn=args.batchnorm,is_training=self.training,
                        act=args.act)

                if 1 in sources:
                    output2 = self._layers('modal1', self.input_data2, 
                        args.dnn_size2, keep_prob=self.keep_prob,
                        bn=args.batchnorm,is_training=self.training,
                        act=args.act)

                if 2 in sources:
                    output3 = self._layers('modal2', self.input_data3, 
                        args.dnn_size3, keep_prob=self.keep_prob,
                        bn=args.batchnorm,is_training=self.training,
                        act=args.act)



            '''' model combination. Two methdos supported:
                A. collaborative prediction: loss = (1 - p1) * log p2 + (1 - p2) * log p1
                B. shared layer
                '''

            if args.collaborative != 0:
                # compute log p and avoid numerical issues
                def compute_logp(logits):
                    logits_max = tf.reduce_max(logits, 1, keep_dims=True)
                    off_logits = logits - logits_max
                    tmp = tf.log(tf.reduce_sum(tf.exp(off_logits), 1, keep_dims=True))
                    logp = off_logits - tmp
                    return logp

            if args.collaborative in  [1, -112]:
                # output 1
                with tf.variable_scope('component1'):
                    logits = self._fully_connected(output1, self.num_classes)
                # output 2
                with tf.variable_scope('component2'):
                    logits2 = self._fully_connected(output2, self.num_classes)
                
                if len(sources) == 3:
                    # output 3
                    with tf.variable_scope('component3'):
                        logits3 = self._fully_connected(output3, self.num_classes)
                # compute log p
                logp = compute_logp(logits)
                logp2 = compute_logp(logits2)
                if len(sources) == 3:
                    logp3 = compute_logp(logits3)
                # compute 1 - p
                one_p = tf.stop_gradient(1 - tf.exp(logp))
                one_p2 = tf.stop_gradient(1 - tf.exp(logp2))
                if len(sources) == 3:
                    one_p3 = tf.stop_gradient(1 - tf.exp(logp3))
                
                # compute loss
                if len(sources) == 3:
                    ce_collaborative = (tf.sqrt(one_p2 * one_p3) * logp + tf.sqrt(one_p *one_p3) * logp2 + tf.sqrt(one_p * one_p2) * logp3)
                else:
                    ce_collaborative = one_p2  * logp + one_p * logp2 

                def construct_mask(logits, K=2, margin=args.margin):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.cast(tf.less(m1m0 , margin), tf.float32) # l1 < l0 + margin
                    m1 = tf.cast(tf.less(-m1m0, margin), tf.float32) # l0 < l1 + margin
                    masks = tf.concat([m0, m1], 1)
                    return tf.stop_gradient(masks)

                masks = construct_mask(-ce_collaborative)
                self.masks = masks


                labels_t = tf.cast(self.labels, tf.float32)
                self.violation = tf.reduce_sum(masks * labels_t)                
                if args.collaborative == 1:
                    loss = -1 * tf.reduce_mean(tf.reduce_sum(ce_collaborative * labels_t, 1))
                elif args.collaborative == -112:
                    loss = -1 * tf.reduce_mean(tf.reduce_sum(ce_collaborative * labels_t * masks, 1))

                self.probs = tf.exp(ce_collaborative)
                self.pred = tf.argmax(ce_collaborative, 1)
                if len(sources) == 3:
                    self.ps = [1 - one_p, 1 - one_p2, 1 - one_p3]
                else:
                    self.ps = [1 - one_p, 1 - one_p2]

                loss_v = []
                loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp * labels_t, 1)))
                loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp2 * labels_t, 1)))
                if len(sources) == 3:
                    loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp3 * labels_t, 1)))
                self.loss_v = loss_v

            elif args.collaborative in [-1, -111]:
                if 2 in sources:
                    outputs_all = [output1, output2, output3]
                else: # sources = [0, 1]
                    outputs_all = [output1, output2]

                L = len(outputs_all)
                assert(L > 1)
                if L == 2:
                    combinations = [[0], [1], [0,1]]
                elif L == 3:
                    combinations = [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
                print('combination {}'.format(combinations))

                logps, one_ps = [], []
                component_outputs = []
                for j in range(len(combinations)):
                    with tf.name_scope('collab{}'.format(j)):
                        with tf.variable_scope('varcollab{}'.format(j)):
                            comb = combinations[j]
                            if len(comb) ==  1:
                                output = outputs_all[comb[0]]
                            else:
                                from attention import merge, merge_attend, my_attention
                                outputs_all_j = [outputs_all[ind] for ind in comb]

                                output = self._merge('addmul' + str(j), outputs_all_j, 
                                    merge_size=args.output_size, bn=args.batchnorm, act='relu')

                            output_size = output.get_shape()[-1].value
                            output = tf.reshape(output, [-1, output_size])

                            # final fully connected layers to output
                            if args.num_fc_layers > 0:
                                if args.hidden_nl == 3:
                                    
                                    output = self._layers('addmul' + str(j), output, 
                                        '300,200', keep_prob=self.keep_prob,
                                        bn=args.batchnorm,is_training=self.training,
                                        act=args.act)

                                else:
                                    output = self._layers('addmul' + str(j), output, 
                                        '500,200', keep_prob=self.keep_prob,
                                        bn=args.batchnorm,is_training=self.training,
                                        act=args.act)


                            component_outputs.append(output)

                            # output
                            output_size = output.get_shape()[-1].value
                            
                            with tf.variable_scope('addmuloutput' + str(j)):
                                logits = self._fully_connected(output, self.num_classes)

                            logp = compute_logp(logits)
                            logps.append(logp)
                            one_p = tf.stop_gradient(1 - tf.exp(logp))
                            one_ps.append(one_p)

                R = range(len(one_ps))
                ce = []
                for j in R:
                    ind0 = list(set(R) - set([j]))
                    if args.mul == 0: # bug....
                        one_p_mean = tf.exp(tf.reduce_mean([tf.log(one_ps[i]) for i in ind0]))
                    elif args.mul == 1:
                        one_p_mean = tf.exp(tf.reduce_mean([tf.log(one_ps[i]) for i in ind0], 0))
                    elif args.mul == 2:
                        one_p_mean = tf.reduce_min([one_ps[i] for i in ind0], 0)                    
                    # one_p_mean = tf.exp(tf.reduce_mean([tf.log(one_ps[i]) for i in ind0]))
                    ce.append(one_p_mean * logps[j])
                ce_collaborative = tf.reduce_sum(ce, 0)
                # ce_collaborative = (tf.sqrt(one_p2 * one_p3) * logp + tf.sqrt(one_p *one_p3) * logp2 + tf.sqrt(one_p * one_p2) * logp3)

                print(ce_collaborative)

                labels_t = tf.cast(self.labels, tf.float32)

                def construct_mask(logits, K=2, margin=args.margin):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.cast(tf.less(m1m0 , margin), tf.float32) # l1 < l0 + margin
                    m1 = tf.cast(tf.less(-m1m0, margin), tf.float32) # l0 < l1 + margin
                    masks = tf.concat([m0, m1], 1)
                    return tf.stop_gradient(masks)


                masks = construct_mask(-ce_collaborative)
                self.masks = masks
                self.violation = tf.reduce_sum(masks * labels_t)
                if args.collaborative in [-111]:
                    loss = tf.reduce_mean(tf.reduce_sum(-ce_collaborative * labels_t * masks, 1))
                else:
                    loss = -1 * tf.reduce_mean(tf.reduce_sum(ce_collaborative * labels_t, 1))


                loss = -1 * tf.reduce_mean(tf.reduce_sum(ce_collaborative * labels_t, 1))

                n_c = n_components = len(component_outputs)

                self.probs = tf.exp(ce_collaborative)
                self.pred = tf.argmax(ce_collaborative, 1)
                self.ps = [1 - one_p for one_p in one_ps]

                loss_v = []
                for logp in logps:
                    loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp * labels_t, 1)))
                self.loss_v = loss_v
            elif args.collaborative in [-3, -4]: # attention for combining probability
                # output 1
                with tf.variable_scope('component1'):
                    softmax_w = tf.get_variable("softmax_w",
                                                [output1.get_shape()[1].value, self.num_classes])
                    softmax_b = tf.get_variable("softmax_b", [self.num_classes])
                logits = tf.matmul(output1, softmax_w) + softmax_b

                # output 2
                with tf.variable_scope('component2'):
                    softmax_w2 = tf.get_variable("softmax_w2",
                                                [output2.get_shape()[1].value, self.num_classes])
                    softmax_b2 = tf.get_variable("softmax_b2", [self.num_classes])
                logits2 = tf.matmul(output2, softmax_w2) + softmax_b2
                
                # output 3
                with tf.variable_scope('component3'):
                    softmax_w3 = tf.get_variable("softmax_w3",
                                                [output3.get_shape()[1].value, self.num_classes])
                    softmax_b3 = tf.get_variable("softmax_b3", [self.num_classes])
                logits3 = tf.matmul(output3, softmax_w3) + softmax_b3
                # compute log p
                logp = compute_logp(logits)
                logp2 = compute_logp(logits2)
                logp3 = compute_logp(logits3)

                # compute 1 - p
                one_p = tf.stop_gradient(1 - tf.exp(logp))
                one_p2 = tf.stop_gradient(1 - tf.exp(logp2))
                one_p3 = tf.stop_gradient(1 - tf.exp(logp3))

                # merge different components to a shared layer with attention
                from attention import merge, my_attention
                att = False if args.late_attention == 0 else True
                sep_attend = True if args.sep_attend == 1 else False
                if args.cross_attend == 1:
                    merge = merge_attend
                elif args.cross_attend == 2:
                    merge = my_attention
                outputs_all = [output1, output2, output3]
                _, alphas_output, _ = merge(outputs_all, 
                    merge_size=args.output_size, attention=att, 
                    attention_size=args.late_attention, sep_attend=sep_attend, hidden_nl=args.hidden_nl)

                print('alphas')
                print(alphas_output)
                print('logp')
                print(logp)

                if args.collaborative == -3 : 
                    logp_comb = alphas_output[0] * logp + alphas_output[1] * logp2 + alphas_output[2] * logp3
                else:
                    p = tf.exp(logp)
                    p2 = tf.exp(logp2)
                    p3 = tf.exp(logp3)
                    logp_comb = tf.log(alphas_output[0] * p + alphas_output[1] * p2 + alphas_output[2] * p3)
                
                labels_t = tf.cast(self.labels, tf.float32)
                loss = -1 * tf.reduce_mean(tf.reduce_sum(logp_comb * labels_t, 1))
                self.probs = tf.exp(logp_comb)
                self.pred = tf.argmax(logp_comb, 1)
                self.ps = [1 - one_p, 1 - one_p2, 1 - one_p3]

                loss_v = []
                loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp * labels_t, 1)))
                loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp2 * labels_t, 1)))
                loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp3 * labels_t, 1)))
                self.loss_v = loss_v

            elif args.collaborative in [-2, -5, -6, -7, -8, -9, -10, -13]: # normlized version
                
                # output 1
                with tf.variable_scope('component1'):
                    softmax_w = tf.get_variable("softmax_w",
                                                [output1.get_shape()[1].value, self.num_classes])
                    softmax_b = tf.get_variable("softmax_b", [self.num_classes])
                logits = tf.matmul(output1, softmax_w) + softmax_b

                # output 2
                with tf.variable_scope('component2'):
                    softmax_w2 = tf.get_variable("softmax_w2",
                                                [output2.get_shape()[1].value, self.num_classes])
                    softmax_b2 = tf.get_variable("softmax_b2", [self.num_classes])
                logits2 = tf.matmul(output2, softmax_w2) + softmax_b2
                
                if len(sources) == 3:
                    # output 3
                    with tf.variable_scope('component3'):
                        softmax_w3 = tf.get_variable("softmax_w3",
                                                    [output3.get_shape()[1].value, self.num_classes])
                        softmax_b3 = tf.get_variable("softmax_b3", [self.num_classes])
                    logits3 = tf.matmul(output3, softmax_w3) + softmax_b3
                
                # compute log p
                logp = compute_logp(logits)
                logp2 = compute_logp(logits2)
                if len(sources) == 3:
                    logp3 = compute_logp(logits3)
                # compute 1 - p
                one_p = tf.stop_gradient(1 - tf.exp(logp))
                one_p2 = tf.stop_gradient(1 - tf.exp(logp2))
                if len(sources) == 3:
                    one_p3 = tf.stop_gradient(1 - tf.exp(logp3))
                
                if args.collaborative == -6:
                    one_p = 1 - tf.exp(logp)
                    one_p2 = 1 - tf.exp(logp2)
                    if len(sources) == 3:
                        one_p3 = 1 - tf.exp(logp3)

                if args.collaborative == -10:
                    # merge different components to a shared layer with attention
                    from attention import merge, my_attention
                    att = False if args.late_attention == 0 else True
                    sep_attend = True if args.sep_attend == 1 else False
                    if args.cross_attend == 1:
                        merge = merge_attend
                    elif args.cross_attend == 2:
                        merge = my_attention
                    if len(sources) == 3:
                        outputs_all = [output1, output2, output3]
                    else:
                        outputs_all = [output1, output2]
                    output, alphas_output, hiddens = merge(outputs_all, 
                        merge_size=args.output_size, attention=att, 
                        attention_size=args.late_attention, sep_attend=sep_attend, hidden_nl=args.hidden_nl)

                # compute loss
                # ce_collaborative = (tf.sqrt(one_p2 * one_p3) * logp + tf.sqrt(one_p *one_p3) * logp2 + tf.sqrt(one_p * one_p2) * logp3)
                # ce_collaborative = tf.reduce_mean([tf.sqrt(one_p2 * one_p3) * logp, tf.sqrt(one_p *one_p3) * logp2, tf.sqrt(one_p * one_p2) * logp3], 0)
                
                # f1 
                def construct_margin_errors(logits, K=self.num_classes):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.nn.relu(m1m0 + 1)
                    m1 = tf.nn.relu(-m1m0 + 1)
                    margin_errors = tf.concat([m0, m1], 1)
                    return margin_errors
                
                def construct_margin_errors0(logits, K=self.num_classes):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.nn.relu(m1m0 + 1)
                    m1 = tf.nn.relu(-m1m0 + 1)
                    margin_errors = tf.concat([m0, m1], 1)
                    return margin_errors

                def construct_margin_errors2(logits, K=self.num_classes):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.square(tf.nn.relu(m1m0 + 1))
                    m1 = tf.square(tf.nn.relu(-m1m0 + 1))
                    margin_errors = tf.concat([m0, m1], 1)
                    return margin_errors

                def construct_margin_errors3(logits, K=self.num_classes): # smooth hinge loss
                    logits_per_class = tf.split(logits, K, 1)
                    m0m1 = logits_per_class[0] - logits_per_class[1]
                    m1m0 = -m0m1
                    m0 = tf.where(tf.less(m0m1, 0), 0.5 - m0m1, 0.5 * tf.square(tf.nn.relu(1 - m0m1)))
                    m1 = tf.where(tf.less(m1m0, 0), 0.5 - m1m0, 0.5 * tf.square(tf.nn.relu(1 - m1m0)))
                    margin_errors = tf.concat([m0, m1], 1)
                    return margin_errors

                def construct_margin_errors4(logits, K=self.num_classes):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.exp(tf.nn.relu(m1m0 + 1)) - 1
                    m1 = tf.exp(tf.nn.relu(-m1m0 + 1)) - 1
                    margin_errors = tf.concat([m0, m1], 1)
                    return margin_errors

                def construct_margin_errors5(logits, K=self.num_classes):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.exp(m1m0)
                    m1 = tf.exp(-m1m0)
                    margin_errors = tf.concat([m0, m1], 1)
                    return margin_errors

                def construct_mask(logits, K=self.num_classes, margin=1):
                    logits_per_class = tf.split(logits, K, 1)
                    m1m0 = logits_per_class[1] - logits_per_class[0]
                    m0 = tf.cast(tf.less(m1m0 , margin), tf.float32) # l1 < l0 + margin
                    m1 = tf.cast(tf.less(-m1m0, margin), tf.float32) # l0 < l1 + margin
                    masks = tf.concat([m0, m1], 1)
                    return tf.stop_gradient(masks)

                if args.collaborative == -2:
                    cme = construct_margin_errors
                elif args.collaborative in [-5, -6, -7]:
                    cme = construct_margin_errors2
                elif args.collaborative in [-8, -9, -10]:
                    # cme = compute_logp
                    cme = construct_margin_errors4
                    cme2 = construct_margin_errors3
                elif args.collaborative in [-13]:
                    cme = compute_logp

                me = cme(logits)
                me2 = cme(logits2)
                if len(sources) == 3:
                    me3 = cme(logits3)

                if len(sources) == 3:
                    if args.collaborative == -7:
                        ce_collaborative = -tf.reduce_mean([tf.minimum(one_p2, one_p3) * me, tf.minimum(one_p, one_p3) * me2, tf.minimum(one_p, one_p2) * me3], 0)
                    elif args.collaborative == -9:
                        ce_collaborative = -tf.reduce_mean([0.5 * (one_p2 + one_p3) * me, 0.5 * (one_p + one_p3) * me2, 0.5 * (one_p + one_p2) * me3], 0)
                    elif args.collaborative == -10:
                        ce_collaborative = -tf.reduce_mean([alphas_output[0] * me, alphas_output[1] * me2, alphas_output[2] * me3], 0)
                    else:
                        ce_collaborative = -tf.reduce_mean([tf.sqrt(one_p2 * one_p3) * me, tf.sqrt(one_p *one_p3) * me2, tf.sqrt(one_p * one_p2) * me3], 0)
                elif len(sources) == 2:
                    if args.collaborative == -13:
                        ce_collaborative = -tf.reduce_mean([one_p2 * me, one_p * me2,], 0)
                # take a softmax
                logp_comb = compute_logp(ce_collaborative)

                # ce_collaborative = tf.nn.softmax(ce_collaborative)

                print(ce_collaborative)

                labels_t = tf.cast(self.labels, tf.float32)

                masks = construct_mask(ce_collaborative)
                self.masks = masks
                self.violation = tf.reduce_sum(masks * labels_t)

                if args.collaborative in [-13]:
                    loss = tf.reduce_mean(tf.reduce_sum(ce_collaborative * labels_t * masks, 1))
                    ce_collaborative = -ce_collaborative
                    logp_comb = compute_logp(ce_collaborative)
                else:
                    margin_errors = cme2(ce_collaborative)
                    loss = tf.reduce_mean(tf.reduce_sum(margin_errors * labels_t, 1))
                # loss = tf.losses.hinge_loss(labels_t, ce_collaborative)
                # loss = tf.reduce_mean(tf.reduce_sum(-ce_collaborative * labels_t, 1))
                # loss = -1 * tf.reduce_mean(tf.reduce_sum(logp_comb * labels_t, 1))
                # self.probs = ce_collaborative
                self.probs = tf.exp(logp_comb)
                self.pred = tf.argmax(ce_collaborative, 1)
                if len(sources) == 3:
                    self.ps = [1 - one_p, 1 - one_p2, 1 - one_p3]
                elif len(sources) == 2:
                    self.ps = [1 - one_p, 1 - one_p2]

                loss_v = []
                loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp * labels_t, 1)))
                loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp2 * labels_t, 1)))
                if len(sources) == 3:
                    loss_v.append(-1 * tf.reduce_mean(tf.reduce_sum(logp3 * labels_t, 1)))
                self.loss_v = loss_v


            elif args.collaborative == 0:
                from attention import merge
                # shared layer + attention before shared layer
                if 2 in sources:
                    outputs_all = [output1, output2, output3]
                else:
                    outputs_all = [output1, output2]

                # log scales of hidden variables
                # mean_outputs, var_outputs = [], []
                # for o in outputs_all:
                #     mean, var = tf.nn.moments(o, axes=[0, 1])
                #     mean_outputs.append(mean)
                #     var_outputs.append(var)

                # virtual losses to monitor training progresses of different components
                loss_v = []
                for i, o in enumerate(outputs_all):
                    with tf.name_scope('virtual{}'.format(i)) and tf.variable_scope('virtual{}'.format(i)):
                        output_size_v = o.get_shape()[-1].value
                        softmax_w = tf.get_variable("softmax_w{}".format(i), [output_size_v, self.num_classes])
                        softmax_b = tf.get_variable("softmax_b{}".format(i), [self.num_classes])    
                        logits_v = tf.matmul(o, softmax_w) + softmax_b
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_v, labels = self.labels))
                        loss_v.append(loss)
                self.loss_v = loss_v

                # merge different components to a shared layer with attention

                # att = False if args.late_attention == 0 else True
                # sep_attend = True if args.sep_attend == 1 else False
                # if args.cross_attend == 1:
                #     merge = merge_attend

                # output, alphas_output, hiddens = merge(outputs_all, 
                #     merge_size=args.output_size, attention=att, 
                #     attention_size=args.late_attention, sep_attend=sep_attend, hidden_nl=args.hidden_nl)

                output = self._merge('additive_merge', outputs_all, 
                    merge_size=args.output_size, bn=args.batchnorm, act='relu')

                # log scales of hidden variables
                # mean_hiddens, var_hiddens = [], []
                # for o in hiddens:
                #     mean, var = tf.nn.moments(o, axes=[0, 1])
                #     mean_hiddens.append(mean)
                #     var_hiddens.append(var)

                output_size = output.get_shape()[-1].value
                output = tf.reshape(output, [-1, output_size])

                # final fully connected layers to output
                if args.num_fc_layers > 0:
                    # output, _ = layers(output, args.batch_size, 
                    #     output_size, args.num_fc_layers, 
                    #     keep_prob=self.keep_prob, bn=args.batchnorm, training=self.training, act=args.act)
                    output = self._layers('additive_fc', output, 
                        '500,200', keep_prob=self.keep_prob,
                        bn=args.batchnorm,is_training=self.training,
                        act=args.act)

                # output
                output_size = output.get_shape()[-1].value
                with tf.variable_scope('additive_finaloutput'):
                    self.logits = self._fully_connected(output, self.num_classes)
                #     softmax_w = tf.get_variable("softmax_w",
                #                                 [output_size, self.num_classes])
                #     softmax_b = tf.get_variable("softmax_b", [self.num_classes])

                # self.logits = tf.matmul(output, softmax_w) + softmax_b
                self.probs = tf.nn.softmax(self.logits)
                self.pred = tf.argmax(self.logits, 1)

                # self.alphas = alphas_output

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels = self.labels))

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels = self.labels))

        with tf.name_scope('cost'):
            self.cost = loss
            self.xent = self.cost
            self.cost += self._decay() ##

        # instrument tensorboard
        # tf.summary.histogram('logits', self.logits)
        if args.collaborative != 0:
            tf.summary.histogram('prob', self.ps)
            for i in range(len(self.ps)):
                tf.summary.histogram('prob{}'.format(i), self.ps[i])
        # else:
        #     if args.late_attention > 0:
        #         tf.summary.histogram('alphas_output', alphas_output)
        #         for i in range(len(alphas_output)):
        #             tf.summary.histogram('alpha_output{}'.format(i), alphas_output[i])

        # tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)
        tf.summary.scalar('train_xent_loss', self.xent)

        if 'mean_outputs' in locals():
            for i, m in enumerate(mean_outputs):
                tf.summary.scalar('mean{}'.format(i), m)
            for i, v in enumerate(var_outputs):
                tf.summary.scalar('var{}'.format(i), v)
        
        if 'mean_hiddens' in locals():
            for i, m in enumerate(mean_hiddens):
                tf.summary.scalar('mean_hidden{}'.format(i), m)
            for i, v in enumerate(var_hiddens):
                tf.summary.scalar('var_hidden{}'.format(i), v)

        if 'loss_v' in locals():
            for i, l in enumerate(loss_v):
                tf.summary.scalar('virtual_loss{}'.format(i), l)

        if not self.is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = [tv for tv in tf.trainable_variables() if 'virtual' not in tv.name]
        if args.grad_clip > 0:        
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                    args.grad_clip)
        else:
            grads = tf.gradients(self.cost, tvars)

        with tf.name_scope('optimizer'):
            if args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif args.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif args.optimizer == 'sgd':
                optimizer = tf.train.SGDOptimizer(self.lr)
            elif args.optimizer == 'mm':
                optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)


        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.train_and_update_op = [self.train_op] + self._bn_update_ops

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            # train_op = optimizer.minimize(loss)        

        if args.collaborative == 0 and len(sources) > 1:
            train_op_v = []
            for i in range(len(outputs_all)):
                tvars_v = [tv for tv in tf.trainable_variables() if 'virtual{}'.format(i) in tv.name ]
                grads_i, _ = tf.clip_by_global_norm(tf.gradients(loss_v[i], tvars_v), args.grad_clip)
                train_op_v.append(optimizer.apply_gradients(zip(grads_i, tvars_v)))
            self.train_op_v = train_op_v
        elif args.collaborative == 0 and len(sources) == 1:
            self.train_op_v = []

    
    
    def _batch_norm(self, name, x, add_ops=True):
        """Batch normalization."""
        with tf.variable_scope(name):
            n_out = x.get_shape()[-1]
            try:
                n_out = int(n_out)
                shape = [n_out]
            except:
                shape = None
            beta = self._weight_variable(
                shape,
                init_method="constant",
                init_param={"val": 0.0},
                name="beta",
                dtype=self.dtype)
            gamma = self._weight_variable(
                shape,
                init_method="constant",
                init_param={"val": 1.0},
                name="gamma",
                dtype=self.dtype)
            normed, ops = batch_norm(
                x,
                self.is_training,
                gamma=gamma,
                beta=beta,
                axes=[0],
                eps=1e-3,
                name="bn_out")
            if add_ops:
                if ops is not None:
                    self._bn_update_ops.extend(ops)
            return normed


    def _weight_variable(self,
                       shape,
                       init_method=None,
                       dtype=tf.float32,
                       init_param=None,
                       wd=None,
                       name=None,
                       trainable=True,
                       seed=0):
        """Wrapper to declare variables. Default on CPU."""
        return weight_variable_cpu(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            wd=wd,
            name=name,
            trainable=trainable,
            seed=seed)

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer"""
        x_shape = x.get_shape()
        d = x_shape[1]
        w = self._weight_variable(
            [d, out_dim],
            init_method="uniform_scaling",
            init_param={"factor": 1.0},
            wd=self.args.wd,
            dtype=self.dtype,
            name="w")
        b = self._weight_variable(
            [out_dim],
            init_method="constant",
            init_param={"val": 0.0},
            name="b",
            dtype=self.dtype)
        return tf.nn.xw_plus_b(x, w, b)

    def _relu(self, name, x):
        return tf.nn.relu(x, name=name)

    def _layers(self, name, x, hidden_size, keep_prob=1.0, 
        bn=0, is_training=True, act='relu', firstlayerdrop=True):
        hidden_sizes = [int(h) for h in hidden_size.split(',')]
        n = len(hidden_sizes)
        if n == 0:
            if keep_prob != 1.0 and firstlayerdrop:
                return tf.nn.dropout(x, keep_prob)
            else:
                return x
        for i, h in enumerate(hidden_sizes):
            name_var = name + '_' + str(i)
            with tf.variable_scope(name_var):
                x = self._fully_connected(x, h)
                if bn > 0:
                    x = self._batch_norm(name_var+'bn', x)
                x = self._relu(name_var+'relu', x)
                if keep_prob != 1.0:
                    x = tf.nn.dropout(x, keep_prob)
        return x

    def _decay(self):
        """L2 weight decay loss."""
        wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        log.info("Weight decay variables")
        [log.info(x) for x in wd_losses]
        log.info("Total length: {}".format(len(wd_losses)))
        if len(wd_losses) > 0:
            return tf.add_n(wd_losses)
        else:
            log.warning("No weight decay variables!")
            return 0.0

    def _merge(self, name, inputs, merge_size=128, bn=0, act=None):
        res = []
        for i, x in enumerate(inputs):
            with tf.variable_scope(name + str(i)):
                x = self._fully_connected(x, merge_size)
            if bn != 0:
                x = self._batch_norm(name + str(i)+'bn', x)
            if act:
                x = self._relu(name+str(i)+'bn', x)
            res.append(x)
        return tf.reduce_mean(res, 0)
