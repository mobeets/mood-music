import json
import os.path
from keras.callbacks import ModelCheckpoint, EarlyStopping

def get_callbacks(args):
    # prepare to save model checkpoints
    chkpt_filename = os.path.join(args.model_dir, args.run_name + '.h5')
    checkpt = ModelCheckpoint(chkpt_filename, monitor='val_loss', save_weights_only=True, save_best_only=True)
    callbacks = [checkpt]
    if args.patience > 0:
        early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=0)
        callbacks.append(early_stop)
    return callbacks

def save_model_in_pieces(model, args):
	# save model structure
    outfile = os.path.join(args.model_dir, args.run_name + '.yaml')
    with open(outfile, 'w') as f:
        f.write(model.to_yaml())
    # save model args
    outfile = os.path.join(args.model_dir, args.run_name + '.json')
    json.dump(vars(args), open(outfile, 'w'))
