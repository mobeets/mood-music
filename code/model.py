import json
import numpy as np
from keras.layers import Input, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Activation, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def get_embed_model(args, do_opt=False):
    args.embedding_dim = 512
    args.hidden_dim = 128

    X = Input(shape=(args.seq_length,), name='X')
    Z = Embedding(output_dim=args.embedding_dim,
            input_dim=args.num_words,
            input_length=args.seq_length)(X)
    zf0 = Dense(args.hidden_dim, activation='relu')(Z)
    zf = Flatten()(zf0)

    if args.do_classify:
        y = Dense(args.n_classes,
            activation='softmax', name='y')(zf)
    else:
        y = Dense(1, activation='sigmoid', name='y')(zf)
    model = Model(X, y)
    if args.do_classify:
        model.compile(optimizer=args.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model.compile(optimizer=args.optimizer,
            loss='binary_crossentropy')
    return model

def get_model(args, do_opt=False):
    n_filters = 16
    kernel_size = 2
    pool_size = 2
    hidden_dim = 8

    X = Input(batch_shape=(args.batch_size,
        args.seq_length, args.original_dim), name='X')
    # X.shape == [100, S, 128]
    if do_opt:
        Xc = Activation('sigmoid')(X)
    else:
        Xc = X

    P1 = Dense(args.original_dim, activation='relu')(Xc)
    if do_opt:
        zf = Flatten()(P1)
    else:
        P1d = Dropout(args.dropout)(P1)
        zf = Flatten()(P1d)

    # P1 = Dense(args.original_dim)(X)
    # P1b = BatchNormalization()(P1)
    # P1a = Activation('relu')(P1b)
    # P1d = Dropout(args.dropout)(P1a)
    # zf = Flatten()(P1d)

    # z = Dense(hidden_dim)(P1d)
    # zb = BatchNormalization()(z)
    # za = Activation('relu')(zb)
    # zd = Dropout(args.dropout)(za)
    # zf = Flatten()(zd)

    if args.do_classify:
        y = Dense(args.n_classes,
            activation='softmax', name='y')(zf)
    else:
        y = Dense(1, activation='sigmoid', name='y')(zf)
    model = Model(X, y)
    if args.do_classify:
        model.compile(optimizer=args.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model.compile(optimizer=args.optimizer,
            loss='binary_crossentropy')
    return model

    C = Conv1D(n_filters, kernel_size,
        input_shape=(args.seq_length, args.original_dim),
        padding='valid', activation='relu', strides=1)(X)
    # C.shape == [100, ~S, n_filters]

    P = MaxPooling1D(pool_size=pool_size)(C)
    # P.shape == [100, ~S/pool_size, n_filters]

    C2 = Conv1D(n_filters/2, kernel_size,
        padding='valid', activation='relu', strides=1)(P)
    P2 = MaxPooling1D(pool_size=pool_size)(C2)
    z = Dense(hidden_dim, activation='relu')(P2)
    zf = Flatten()(z)
    if args.do_classify:
        y = Dense(args.n_classes,
                activation='softmax', name='y')(zf)
    else:
        y = Dense(1, activation='sigmoid', name='y')(zf)

    model = Model(X, y)
    if args.do_classify:
        model.compile(optimizer=args.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model.compile(optimizer=args.optimizer,
            loss='binary_crossentropy')
    return model

class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

def load_model(model_file, do_opt=False):
    margs = json.load(open(model_file.replace('.h5', '.json')))
    margs = AttrDict(margs)
    margs.batch_size = 1
    margs.dropout = 0.0
    if margs.do_embed:
        model = get_embed_model(margs, do_opt=do_opt)
    else:
        model = get_model(margs, do_opt=do_opt)
    model.load_weights(model_file)
    return model, margs
