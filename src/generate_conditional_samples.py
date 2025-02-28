#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    length=None,
    max_context_length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    models_dir='models',
    checkpoint_dir='checkpoint',
    run_name='117M',
    prompt_path=None,
    out_path=None
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :max_context_length=None : Number of tokens to use as context, affects
     how much we'll generate each iteration
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context_tokens = []
    with open(prompt_path, 'r') as fp:
        raw_text = fp.read()
        if not raw_text:
            print('Prompt should not be empty!')
            return
        context_tokens = enc.encode(raw_text)

    if length is None:
        # length = hparams.n_ctx // 2
        length = hparams.n_ctx - len(context_tokens)
    # elif len(context_tokens) > hparams.n_ctx - length:
    #     raise ValueError("Can't get samples longer than window size - context: %s" % hparams.n_ctx - len(context_tokens))
    # elif length > hparams.n_ctx:
    #     raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    print('using context of length: %d' % len(context_tokens))

    if max_context_length is None:
        max_context_length = hparams.n_ctx // 2
    elif max_context_length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    if len(context_tokens) > max_context_length:
        print('context is too long! will be truncated...')

    max_block_length = hparams.n_ctx - max_context_length

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        ckpt = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, run_name))

        generated = 0
        all_text = []
        for _ in range(nsamples):

            generated_tokens = []
            context_buffer = None
            while len(generated_tokens) < length:
                if not context_buffer:
                    context_buffer = context_tokens[-hparams.n_ctx:]
                context_length = min(max_context_length, len(context_buffer))
                block_length = hparams.n_ctx - context_length
                if len(generated_tokens) + block_length > length:
                    block_length = length - len(generated_tokens)
                    context_length = hparams.n_ctx - block_length

                print('generating block of %d tokens with context:\n%s' % (block_length, enc.decode(context_buffer[-context_length:])))
                context = tf.placeholder(tf.int32, [1, None])
                output = sample.sample_sequence(
                    hparams=hparams,
                    length=block_length,
                    context=context,
                    batch_size=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                out = sess.run(output, feed_dict={
                    context: [context_buffer[-context_length:]]
                })[0, -block_length:]
                print('generated:\n%s (%d)' % (enc.decode(out), len(out)))

                if len(context_buffer) < hparams.n_ctx:
                    context_buffer.extend(out) # should be at n_ctx now...
                else:
                    # rotate context, newly generated context at the end
                    context_buffer[:context_length] = context_buffer[-context_length:]
                    context_buffer[-block_length:] = out
                generated_tokens.extend(out)
                print('generated %d of %d tokens' % (len(generated_tokens), length))

            generated += 1
            text = enc.decode(context_tokens)
            text += enc.decode(generated_tokens)
            separator = '=' * 40 + ' SAMPLE ' + str(generated) + ' ' + '=' * 40 + '\n'
            print(separator + text)
            all_text.append(separator + text)
        if out_path:
            with open(out_path, 'w') as fp:
                fp.write('\n'.join(all_text))


if __name__ == '__main__':
    fire.Fire(interact_model)
