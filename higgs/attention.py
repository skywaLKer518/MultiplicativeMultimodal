import tensorflow as tf

def init_attention(hidden_size, attention_size):
    # W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], 
    #     stddev=0.1))
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], 
        stddev=tf.sqrt(2.0 / (hidden_size + attention_size))))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=tf.sqrt(2.0 / (attention_size))))
    return W_omega, b_omega, u_omega    

def init_attention2(hidden_size, attention_size, K):
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], 
        stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size, K], stddev=0.1))
    return W_omega, b_omega, u_omega    

def merge2(inputs, x, merge_size=128, attention=False, attention_size=256, sep_attend=True, return_alphas=True):

    """
    merge with or without attention
    Args:
        inputs:
            a list of tensors of shape [batch_size, d1]
        x: 
            a tensor of shape [batch_size, d2]
        merge_size: the output tensor size. transformation matrix is used to transform input
        attention: whether to use merge attention or not
        sep_attend: if true, use separate attention module to attend

    Returns:
        output a tensor of shape [batch_size, merge_size]
    """

    w_x = tf.Variable(tf.random_normal([x.shape[1].value, merge_size], stddev=0.1))
    b_x = tf.Variable(tf.random_normal([merge_size], stddev=0.1))
    z = tf.matmul(x, w_x) + b_x

    if not attention:
        return [tf.reduce_mean([e, z], 0) for e in inputs], 0
    else:
        W_omegas, b_omegas, u_omegas = [], [], []
        inds = {}
        if sep_attend:
            for i in range(2):
                inds[i] = i
                w, b, u = init_attention(merge_size, attention_size)
                W_omegas.append(w)
                b_omegas.append(b)
                u_omegas.append(u)
        else:
            w, b, u = init_attention(merge_size, attention_size)
            W_omegas.append(w)
            b_omegas.append(b)
            u_omegas.append(u)
            for i in range(2):
                inds[i] = 0
        
        pre_sm = [] # a list of B by 1
        for i, h in enumerate(inputs):
            idx = inds[0]
            pre_sm.append(tf.matmul(tf.tanh(tf.matmul(h, 
                W_omegas[idx]) + tf.reshape(
                b_omegas[idx], [1, -1])), tf.reshape(u_omegas[idx], [-1, 1])))
        idx2 = inds[1]
        pre_sm2 = tf.matmul(tf.tanh(tf.matmul(z, 
                W_omegas[idx2]) + tf.reshape(
                b_omegas[idx2], [1, -1])), tf.reshape(u_omegas[idx2], [-1, 1]))
        # B by 1


        alphas = [tf.nn.softmax(tf.concat([presm, pre_sm2], 1)) for presm in pre_sm]
        # a list of B by 2 tensors
        alphas = [tf.split(a, 2, 1) for a in alphas]

        outputs = [tf.reduce_mean([e * a[0], z * a[1]], 0) for e, a in zip(inputs, alphas)]
        
        print('len of outputs:')
        print(len(outputs))
        print('the first elementof outputs')
        print(outputs[0])

        if return_alphas:
            return outputs, alphas
        else:
            return outputs, 0


def merge_attend(inputs, merge_size=128, attention=False, attention_size=256, sep_attend=True, return_alphas=True, hidden_nl=0):
    W_projs, B_projs, hiddens = [], [], []
    for x in inputs:
        W_projs.append(tf.Variable(tf.random_normal([x.shape[1].value, merge_size], stddev=0.1)))
        B_projs.append(tf.Variable(tf.random_normal([merge_size], stddev=0.1)))
        if hidden_nl > 0:
            hiddens.append(tf.tanh(tf.matmul(x, W_projs[-1]) + B_projs[-1]))
        else:
            hiddens.append(tf.matmul(x, W_projs[-1]) + B_projs[-1])


    hidden_concat = tf.concat(hiddens, 1)
    w,b,u = init_attention2(merge_size* len(hiddens), attention_size, len(hiddens))
    sm = tf.nn.softmax(tf.matmul(tf.tanh(tf.matmul(hidden_concat, w) + b), u)) # batch by K
    alphas = tf.split(sm, len(hiddens), 1)
    print(type(alphas), alphas)

    outputs = []
    for i in range(len(inputs)):
        outputs.append(hiddens[i] * alphas[i])
    print('len of outputs:')
    print(len(outputs))
    print('the first element of outputs')
    print(outputs[0])

    return tf.reduce_mean(outputs, 0), alphas, hiddens


def merge(inputs, merge_size=128, attention=False, attention_size=256, sep_attend=True, return_alphas=True, hidden_nl=0):

    """
    merge with or without attention
    Args:
        inputs:
            a list of tensors of shape [batch_size, ?]
        merge_size: the output tensor size. transformation matrix is used to transform input
        attention: whether to use merge attention or not
        sep_attend: if true, use separate attention module to attend

    Returns:
        output a tensor of shape [batch_size, merge_size]
    """

    W_projs, B_projs, hiddens = [], [], []
    for x in inputs:
        # W_projs.append(tf.Variable(tf.random_normal([x.shape[1].value, merge_size], stddev=0.1)))
        W_projs.append(tf.Variable(tf.random_normal([x.shape[1].value, merge_size], stddev=tf.sqrt(2.0 / (x.shape[1].value + x.shape[0].value)))))
        # B_projs.append(tf.Variable(tf.random_normal([merge_size], stddev=0.1)))
        B_projs.append(tf.Variable(tf.random_normal([merge_size], stddev=0.1)))
        if hidden_nl == 1:
            hiddens.append(tf.tanh(tf.matmul(x, W_projs[-1]) + B_projs[-1]))
        elif hidden_nl == 2:
            hiddens.append(tf.nn.relu(tf.matmul(x, W_projs[-1]) + B_projs[-1]))
        else:
            hiddens.append(tf.matmul(x, W_projs[-1]) + B_projs[-1])
    
    if not attention:
        return tf.reduce_mean(hiddens, 0), [tf.constant(1.0)] * len(hiddens), hiddens
    else:
        W_omegas, b_omegas, u_omegas = [], [], []
        inds = {}
        if sep_attend:
            for i, x in enumerate(inputs):
                inds[i] = i
                w, b, u = init_attention(merge_size, attention_size)
                W_omegas.append(w)
                b_omegas.append(b)
                u_omegas.append(u)

        else:
            w, b, u = init_attention(merge_size, attention_size)
            W_omegas.append(w)
            b_omegas.append(b)
            u_omegas.append(u)
            for i in range(len(inputs)):
                inds[i] = 0
        
        pre_sm = [] # a list of B by 1
        for i, h in enumerate(hiddens):
            idx = inds[i]
            pre_sm.append(tf.matmul(tf.tanh(tf.matmul(hiddens[i], 
                W_omegas[idx]) + tf.reshape(
                b_omegas[idx], [1, -1])), tf.reshape(u_omegas[idx], [-1, 1])))

        alphas = tf.split(tf.nn.softmax(tf.concat(pre_sm, 1)), len(hiddens), 1)
        print(type(alphas), alphas)

        outputs = []
        for i in range(len(inputs)):
            outputs.append(hiddens[i] * alphas[i])
        print('len of outputs:')
        print(len(outputs))
        print('the first elementof outputs')
        print(outputs[0])

        if return_alphas:
            return tf.reduce_mean(outputs, 0), alphas, hiddens
        else:
            return tf.reduce_mean(outputs, 0), 0, hiddens

def my_attention(inputs, merge_size=0, attention=True, attention_size=256, sep_attend=True, return_alphas=True, hidden_nl=0):

    """
    Args:
        inputs:
            a list of tensors of shape [batch_size, ?]
        merge_size: the output tensor size. transformation matrix is used to transform input
        attention: whether to use merge attention or not
        sep_attend: if true, use separate attention module to attend

    Returns:
        output a tensor of shape [batch_size, merge_size]
    """

    W_projs, B_projs, hiddens = [], [], []
    # for x in inputs:
    #     # W_projs.append(tf.Variable(tf.random_normal([x.shape[1].value, merge_size], stddev=0.1)))
    #     W_projs.append(tf.Variable(tf.random_normal([x.shape[1].value, merge_size], stddev=tf.sqrt(2.0 / (x.shape[1].value + x.shape[0].value)))))
    #     # B_projs.append(tf.Variable(tf.random_normal([merge_size], stddev=0.1)))
    #     B_projs.append(tf.Variable(tf.random_normal([merge_size], stddev=0.1)))
    #     if hidden_nl == 1:
    #         hiddens.append(tf.tanh(tf.matmul(x, W_projs[-1]) + B_projs[-1]))
    #     elif hidden_nl == 2:
    #         hiddens.append(tf.nn.relu(tf.matmul(x, W_projs[-1]) + B_projs[-1]))
    #     else:
    #         hiddens.append(tf.matmul(x, W_projs[-1]) + B_projs[-1])
    
    
    W_omegas, b_omegas, u_omegas = [], [], []
    inds = {}

    for i, x in enumerate(inputs):
        inds[i] = i
        w, b, u = init_attention(x.shape[1].value, attention_size)
        W_omegas.append(w)
        b_omegas.append(b)
        u_omegas.append(u)


    pre_sm = [] # a list of B by 1
    for i, h in enumerate(inputs):
        idx = inds[i]
        pre_sm.append(tf.matmul(tf.tanh(tf.matmul(inputs[i], 
            W_omegas[idx]) + tf.reshape(
            b_omegas[idx], [1, -1])), tf.reshape(u_omegas[idx], [-1, 1])))

    alphas = tf.split(tf.nn.softmax(tf.concat(pre_sm, 1)), len(inputs), 1)
    print(type(alphas), alphas)

    
    return None, alphas, None


def attention(inputs, attention_size, time_major=True, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.

    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        # inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        inputs = tf.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
