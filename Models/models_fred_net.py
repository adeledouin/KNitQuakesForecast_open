import numpy as np
from keras.layers import (
    Conv1D, LeakyReLU, BatchNormalization, Dropout, Lambda, Concatenate, Add,
    CuDNNLSTM, LSTM, Bidirectional, Input, Dense, Activation, dot, concatenate
)
from keras.models import Model
from keras.regularizers import l1, l2
from dna_decipher.generate_data import DNA
dna = DNA()

def residual_network(x, reg, cardinality):
    """
    ResNeXt by default. For ResNet set cardinality = 1.

    References:
    - https://blog.waya.ai/deep-residual-learning-9610bb62c355
    """
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _dilation_rate):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return Conv1D(nb_channels, 3, dilation_rate=_dilation_rate, padding='same')(y)

        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = Lambda(lambda z: z[:, :, j * _d:j * _d + _d])(y)
            groups.append(Conv1D(_d, 3, dilation_rate=_dilation_rate,
                                 padding='same', kernel_regularizer=l2(reg))(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = Concatenate()(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _dilation_rate=1, _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:

        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = Conv1D(nb_channels_in, 1, padding='same', kernel_regularizer=l2(reg))(y)
        y = add_common_layers(y)
        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _dilation_rate=_dilation_rate)
        y = add_common_layers(y)
        y = Conv1D(nb_channels_out, 1, padding='same', kernel_regularizer=l2(reg))(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = BatchNormalization()(y)
        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut:
            shortcut = Conv1D(nb_channels_out, 1, dilation_rate=_dilation_rate,
                              padding='same', kernel_regularizer=l2(reg))(shortcut)
            shortcut = BatchNormalization()(shortcut)
            y = Add()([shortcut, y])
        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = LeakyReLU()(y)
        return y

    # conv1
    x = Conv1D(256, 7, padding='same')(x)
    x = add_common_layers(x)
    # conv2
    project_shortcut = True
    for i in range(5):
        # project_shortcut = True if i == 0 else False
        x = residual_block(x, 256, 512, _project_shortcut=project_shortcut, _dilation_rate=2**i)
        x = Dropout(0.1)(x)
    # conv3
    for i in range(5):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        x = residual_block(x, 512, 512, _project_shortcut=project_shortcut, _dilation_rate=2**i)
        x = Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(256, return_sequences=True, kernel_regularizer=l2(reg)))(x)
    x = LeakyReLU()(x)
    return x


def build_fred_net(batch_size, histo_length, seq_length, n_oligos, n_units, reg, cardinality):
    """Build Fred Net.

    Parameters
    ----------
    - batch_size
    - histo_length
    - seq_length
    - n_oligos
    - n_units : number of cells to create in the encoder and decoder models.
    - reg : L2 regularization
    - cardinality : cardinalty in residual_network.

    Note:  we assume by design that
    - encoder_input_shape  = (batch_size, histo_length, n_oligos)
    - decoder_input_shape  = (batch_size, seq_length, n_bases_oos)
    - encoder_target_shape = (batch_size, histo_length, n_bases_oos)
    - decoder_target_shape = (batch_size, seq_length, n_bases)

    Returns
    -------
    - model: model that can be trained given
        [encoder_input, decoder_input], [encoder_target, decoder_target]

    References:
    - https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
    """
    # shapes
    encoder_input_shape = (batch_size, histo_length, n_oligos)
    decoder_input_shape = (batch_size, seq_length, 5)
    encoder_output_size = 5
    decoder_output_size = 4
    # input tensors
    encoder_input = Input(batch_shape=encoder_input_shape)
    decoder_input = Input(batch_shape=decoder_input_shape)
    # encoder interm and output
    encoder_interm = residual_network(encoder_input, reg, cardinality)
    encoder_output = Dense(encoder_output_size)(encoder_interm)
    encoder_output = Activation('softmax')(encoder_output)

    # encoder
    encoder = Bidirectional(CuDNNLSTM(
        n_units,
        return_sequences=True,
        kernel_regularizer=l2(reg)
    ))(encoder_interm)
    encoder = LeakyReLU()(encoder)
    encoder, state_h, state_c = CuDNNLSTM(
        n_units, return_sequences=True, return_state=True
    )(encoder)

    # decoder
    decoder = CuDNNLSTM(n_units, return_sequences=True)(
        decoder_input, initial_state=[state_h, state_c]
    )
    decoder = LeakyReLU()(decoder)
    decoder = CuDNNLSTM(n_units, return_sequences=True)(decoder)
    decoder = LeakyReLU()(decoder)
    # attention mechanism
    attention = dot([decoder, encoder], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, encoder], axes=[2, 1])
    decoder_combined_context = concatenate([context, decoder])
    # decoder output
    decoder_output = CuDNNLSTM(n_units, return_sequences=True)(
        decoder_combined_context
    )
    decoder_output = LeakyReLU()(decoder_output)
    decoder_output = CuDNNLSTM(decoder_output_size, return_sequences=True)(
        decoder_output
    )
    decoder_output = Activation('softmax')(decoder_output)
    # model to train
    model = Model(
        inputs=[encoder_input, decoder_input],
        outputs=[encoder_output, decoder_output]
    )
    return model

def fred_net_predict_from_peaks(peaks, model, data, batch_size, return_encoder_output = False):
    """Predict DNA sequence from peaks using fred net model.

    Note:  the tensors shape in fred net model must be consistent with data
    and batch_size

    Returns
    -------
    - seq_pred: predicted sequence
    - seq_profile_pred : predicted sequence in the histograms coordinates
        Only when return_encoder_output = True
    """
    sample = 0
    # encoder_input : peaks
    encoder_input = np.zeros((batch_size, data.histo_length, data.n_oligos))
    encoder_input[sample] = peaks
    # decoder_input : initialize with OOS
    decoder_input = np.zeros((batch_size, data.seq_length, dna.n_bases_oos))
    decoder_input[sample,0,0] = 1
    # loop over sequence position
    for x in range(1, data.seq_length):
        encoder_pred, decoder_pred = model.predict([encoder_input, decoder_input], batch_size=batch_size)
        proba = decoder_pred[sample,x-1,:]
        nuc = dna.bases[proba.argmax()]
        decoder_input[sample,x] = dna.indicator(nuc, dna.bases_oos)
    # predicted sequence
    seq_pred = dna.indicator2seq(decoder_pred[sample], dna.bases)
    if return_encoder_output:
        seq_profile_pred = dna.indicator2seq(encoder_pred[sample], ["-"] + dna.bases)
        return seq_pred, seq_profile_pred
    return seq_pred

def fred_net_predict_from_batch(encoder_input, model, data, batch_size, return_encoder_output = False):
    """Predict DNA sequence from peaks using fred net model, for all samples
    in a batch.

    Note:  the tensors shape in fred net model must be consistent with data
    and batch_size

    Returns
    -------
    - seq_pred: predicted sequences
    - seq_profile_pred : predicted sequences in the histograms coordinates
        Only when return_encoder_output = True
    """
    # decoder_input : initialize with OOS
    decoder_input = np.zeros((batch_size, data.seq_length, dna.n_bases_oos))
    decoder_input[:,0,0] = 1
    # loop over sequence position
    for x in range(1, data.seq_length):
        encoder_pred, decoder_pred = model.predict([encoder_input, decoder_input], batch_size=batch_size)
        for sample in range(batch_size):
            proba = decoder_pred[sample,x-1,:]
            nuc = dna.bases[proba.argmax()]
            decoder_input[sample,x] = dna.indicator(nuc, dna.bases_oos)
    # predicted sequence
    seq_pred = [
        dna.indicator2seq(decoder_pred[sample], dna.bases)
        for sample in range(batch_size)
    ]
    if return_encoder_output:
        seq_profile_pred = [
            dna.indicator2seq(encoder_pred[sample], ["-"] + dna.bases)
            for sample in range(batch_size)
        ]
        return seq_pred, seq_profile_pred
    return seq_pred
