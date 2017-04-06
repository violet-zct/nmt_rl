import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
import dynet as dy
import numpy as np
import argparse
from dynet import *
import copy

from nltk.translate.bleu_score import corpus_bleu
import time
from utils import *


class EncoderDecoder:
    def __init__(self, config):
        self.model = model = dy.Model()
        self.config = config
        self.src_voc_size = config['src_voc_size']
        self.tgt_voc_size = config['tgt_voc_size']
        self.emb_size = config['emb_size']
        self.hidden_dim = config['hidden_dim']
        self.att_dim = config["att_dim"]
        self.src_emb = model.add_lookup_parameters((self.src_voc_size, self.emb_size))
        self.tgt_emb = model.add_lookup_parameters((self.tgt_voc_size, self.emb_size))
        self.enc_forward_rnn = GRUBuilder(config['layers'], self.emb_size, self.hidden_dim, model)
        self.enc_backward_rnn = GRUBuilder(config['layers'], self.emb_size, self.hidden_dim, model)

        self.W_init_state = model.add_parameters((self.hidden_dim, self.hidden_dim*2))
        self.b_init_state = model.add_parameters((self.hidden_dim, 1))
        self.b_init_state.zero()
        #self.dec_rnn = GRUBuilder(config['layers'], self.emb_size, self.hidden_dim, model)
        self.dec_rnn = GRUBuilder(config['layers'], self.emb_size + self.hidden_dim*2, self.hidden_dim, model)
        # read out parameters

        if config["concat_readout"]:
            # self.W_readout = model.add_parameters((self.tgt_voc_size, self.hidden_dim * 3))
            # self.b_readout = model.add_parameters((self.tgt_voc_size))
            self.W_readout = model.add_parameters((self.emb_size, self.hidden_dim * 3))
            self.b_readout = model.add_parameters((self.emb_size))
            self.b_readout.zero()
        else:
            self.W_logit_cxt = model.add_parameters((self.emb_size, self.hidden_dim * 2))
            self.W_logit_input = model.add_parameters((self.emb_size, self.emb_size))
            self.W_logit_hid = model.add_parameters((self.emb_size, self.hidden_dim))
            self.b_logit_readout = model.add_parameters((self.emb_size, 1))
            self.b_logit_readout.zero()

        # attention
        self.W_att_hidden = model.add_parameters((self.att_dim, self.hidden_dim))
        self.W_att_cxt = model.add_parameters((self.att_dim, self.hidden_dim*2))
        self.V_att = model.add_parameters((1, self.att_dim))
        #self.b_att = model.add_parameters((1,))
        #self.b_att.zero()

        self.softmax_W = model.add_parameters((self.tgt_voc_size, self.emb_size))
        self.softmax_b = model.add_parameters((self.tgt_voc_size, ))
        self.softmax_b.zero()
        self.SOS = 1
        self.EOS = 2

        # self.src_id_to_word = src_id_to_word
        # self.tgt_id_to_word = tgt_id_to_word

    def save(self):
        self.model.save("obj/" + config["model_name"] + "_params.bin")

    def load(self):
        self.model.load("obj/" + config["model_name"] + "_params.bin")

    def transpose_input(self, seq):
        max_len = max([len(sent) for sent in seq])
        seq_pad = []
        seq_mask = []
        for i in range(max_len):
            pad_temp = [sent[i] if i < len(sent) else self.EOS for sent in seq]
            mask_temp = [1 if i < len(sent) else 0 for sent in seq]
            seq_pad.append(pad_temp)
            seq_mask.append(mask_temp)
        return seq_pad, seq_mask

    def encode(self, src_seq):
        # src_seq is a batch with the same length
        dy.renew_cg()
        src_pad, src_mask = self.transpose_input(src_seq)
        wemb = [dy.lookup_batch(self.src_emb, wids) for wids in src_pad] # (time_step, emb_size, batch_size)
        wemb_r = wemb[::-1]
        fwd_vectors = self.enc_forward_rnn.initial_state().transduce(wemb)
        bwd_vectors = self.enc_backward_rnn.initial_state().transduce(wemb_r)[::-1]

        seq_enc = [dy.concatenate([fwd_v, bwd_v]) for (fwd_v, bwd_v) in zip(fwd_vectors, bwd_vectors)]
        return seq_enc # (time_step, hid_size*2, batch_size)

    def attention(self, encoding, hidden, batch_size):
        W_att_cxt = dy.parameter(self.W_att_cxt)
        W_att_hid = dy.parameter(self.W_att_hidden)
        V_att = dy.parameter(self.V_att)
        #b_att = dy.parameter(self.b_att)

        if self.config["for_loop_att"]:
            temp = W_att_hid * hidden
            enc_list = [V_att * dy.tanh(dy.affine_transform([temp, W_att_cxt, cxt])) for cxt in encoding]
            # enc_list = [V_att * dy.tanh(W_att_cxt * cxt + temp) + b_att for cxt in encoding]
            att_weights = dy.softmax(dy.concatenate(enc_list)) # (time_step, batch_size)
            att_ctx = dy.esum([h_t * att_t for (h_t, att_t) in zip(encoding, att_weights)])
            return att_ctx, att_weights

        enc_seq = dy.concatenate_cols(encoding) # (dim, time_step, batch_size)
        att_mlp = dy.tanh(dy.colwise_add(W_att_cxt * enc_seq, W_att_hid * hidden))

        # att_weights= dy.reshape(V_att * att_mlp + b_att, (len(encoding), ), batch_size)
        att_weights= dy.reshape(V_att * att_mlp, (len(encoding), ), batch_size)

        att_weights = dy.softmax(att_weights) # (time_step, batch_size)
        att_ctx = enc_seq * att_weights
        # print att_ctx.npvalue().shape
        return att_ctx, att_weights

    def decode_loss(self, encoding, tgt_seq):
        W_init_state = dy.parameter(self.W_init_state)
        b_init_state = dy.parameter(self.b_init_state)

        if self.config["concat_readout"]:
            W_readout = dy.parameter(self.W_readout)
            b_readout = dy.parameter(self.b_readout)
        else:
            W_logit_cxt = dy.parameter(self.W_logit_cxt)
            W_logit_input = dy.parameter(self.W_logit_input)
            W_logit_hid = dy.parameter(self.W_logit_hid)
            b_logit_readout = dy.parameter(self.b_logit_readout)

        softmax_w = dy.parameter(self.softmax_W)
        softmax_b = dy.parameter(self.softmax_b)

        # tgt sequence starts from <S>, ends at <\S>
        batch_size = len(tgt_seq)

        init_state = dy.tanh(dy.affine_transform([b_init_state, W_init_state, encoding[-1]]))
        # init_state = dy.tanh(W_init_state * encoding[-1] + b_init_state)
        dec_state = self.dec_rnn.initial_state([init_state]) # not sure
        
        # # TODO: not sure about it, concatenate column vectors?
        # zero_emb = [[0.0]*len(tgt_seq) for _ in range(self.emb_size)]
        tgt_pad, tgt_mask = self.transpose_input(tgt_seq)
        max_len = max([len(sent) for sent in tgt_seq])
        att_ctx = dy.vecInput(self.hidden_dim * 2)
        # shifted_tgt_emb = dy.concatenate(zero_emb + tgt_emb)
        # dec_states = self.dec_rnn.initial_state(enc_rep).transduce(shifted_tgt_emb)
        losses = []
        for i in range(max_len - 1):
            input_t = dy.lookup_batch(self.tgt_emb, tgt_pad[i])
            dec_state = dec_state.add_input(dy.concatenate([input_t, att_ctx]))
            ht = dec_state.output()
            att_ctx, att_weights = self.attention(encoding, ht, batch_size)
            if config["concat_readout"]:
                read_out = dy.tanh(dy.affine_transform([b_readout, W_readout, dy.concatenate([ht, att_ctx])]))
            else:
                read_out = dy.tanh(W_logit_cxt * att_ctx + W_logit_hid * ht + W_logit_input * input_t + b_logit_readout)
            if config["dropout"] > 0:
                read_out = dy.dropout(read_out, config["dropout"])
            prediction = softmax_w * read_out + softmax_b

            loss = dy.pickneglogsoftmax_batch(prediction, tgt_pad[i+1])

            if 0 in tgt_mask[i+1]:
                mask_expr = dy.inputVector(tgt_mask[i+1])
                mask_expr = dy.reshape(mask_expr, (1,), batch_size)
                loss = loss * mask_expr

            losses.append(loss)

        loss = dy.esum(losses)
        loss = dy.sum_batches(loss) / batch_size

        return loss

    def gen_samples(self, src_seq, max_len=30):
        encoding = self.encode([src_seq])
        beam_size = self.config["beam_size"]

        W_init_state = dy.parameter(self.W_init_state)
        b_init_state = dy.parameter(self.b_init_state)

        if self.config["concat_readout"]:
            W_readout = dy.parameter(self.W_readout)
            b_readout = dy.parameter(self.b_readout)
        else:
            W_logit_cxt = dy.parameter(self.W_logit_cxt)
            W_logit_input = dy.parameter(self.W_logit_input)
            W_logit_hid = dy.parameter(self.W_logit_hid)
            b_logit_readout = dy.parameter(self.b_logit_readout)


        softmax_w = dy.parameter(self.softmax_W)
        softmax_b = dy.parameter(self.softmax_b)

        live = 1
        dead = 0

        final_scores = []
        final_samples = []

        scores = np.zeros(live)
        dec_states = [self.dec_rnn.initial_state([dy.tanh(dy.affine_transform([b_init_state, W_init_state, encoding[-1]]))])]
        att_ctxs = [dy.vecInput(self.hidden_dim * 2)]
        samples = [[self.SOS]]

        for ii in range(max_len):
            cand_scores = []
            for k in range(live):
                y_t = dy.lookup(self.tgt_emb, samples[k][-1])
                dec_states[k] = dec_states[k].add_input(dy.concatenate([y_t, att_ctxs[k]]))
                h_t = dec_states[k].output()
                att_ctx, att_weights = self.attention(encoding, h_t, 1)
                att_ctxs[k] = att_ctx
                if config["concat_readout"]:
                    read_out = dy.tanh(dy.affine_transform([b_readout, W_readout, dy.concatenate([h_t, att_ctx])]))
                else:
                    read_out = dy.tanh(W_logit_cxt * att_ctx + W_logit_hid * h_t + W_logit_input * y_t + b_logit_readout)
                prediction = dy.log_softmax(softmax_w * read_out + softmax_b).npvalue()
                cand_scores.append(scores[k] - prediction)

            cand_scores = np.concatenate(cand_scores).flatten()
            ranks = cand_scores.argsort()[:(beam_size - dead)]

            cands_indices = ranks / self.tgt_voc_size
            cands_words = ranks % self.tgt_voc_size
            cands_scores = cand_scores[ranks]

            new_scores = []
            new_dec_states = []
            new_att_ctxs = []
            new_samples = []
            for idx, [bidx, widx] in enumerate(zip(cands_indices, cands_words)):
                new_scores.append(copy.copy(cands_scores[idx]))
                new_dec_states.append(dec_states[bidx])
                new_att_ctxs.append(att_ctxs[bidx])
                new_samples.append(samples[bidx] + [widx])

            scores = []
            dec_states = []
            att_ctxs = []
            samples = []

            for idx, sample in enumerate(new_samples):
                if new_samples[idx][-1] == self.EOS:
                    dead += 1
                    final_samples.append(new_samples[idx])
                    final_scores.append(new_scores[idx])
                else:
                    dec_states.append(new_dec_states[idx])
                    att_ctxs.append(new_att_ctxs[idx])
                    samples.append(new_samples[idx])
                    scores.append(new_scores[idx])
            live = beam_size - dead

            if dead == beam_size:
                break

        if live > 0:
            for idx in range(live):
                final_scores.append(scores[idx])
                final_samples.append(samples[idx])

        return final_scores, final_samples

    def get_encdec_loss(self, src_seqs, tgt_seqs):
        # src_seq and tgt_seq are batches that are not padded to the same length
        encoding = self.encode(src_seqs)
        loss = self.decode_loss(encoding, tgt_seqs)
        return loss


def train(args, config):
    print >>sys.stderr, "Configurations: ", args
    src_vocab, src_id_to_words, src_train = get_vocab(args.train_src, args.src_vocab_size)
    tgt_vocab, tgt_id_to_words, tgt_train = get_vocab(args.train_tgt, args.tgt_vocab_size)

    print >>sys.stderr, "Size of the source and target vocabulary: ", len(src_vocab), len(tgt_vocab)
    config['src_voc_size'] = len(src_vocab)
    config['tgt_voc_size'] = len(tgt_vocab)
    src_dev = get_data(args.dev_src, src_vocab)
    tgt_dev = get_data(args.dev_tgt, tgt_vocab)

    train_data = zip(src_train, tgt_train)
    dev_data = zip(src_dev, tgt_dev)

    nmt_model = EncoderDecoder(config)
    trainer = dy.AdamTrainer(nmt_model.model)

    epochs = 10
    updates = 0
    valid_history = []
    bad_counter = 0
    total_loss = total_examples = 0
    start_time = time.time()
    for epoch in range(epochs):
        for (src_batch, tgt_batch) in data_iterator(train_data, args.batch_size):
            updates += 1
            bs = len(src_batch)

            if updates % args.valid_freq == 0:
                print >>sys.stderr, "#################  Evaluating bleu score on the validation corpus ##############"
                begin_time = time.time()
                bleu_score, translation = translate(nmt_model, dev_data, src_id_to_words, tgt_id_to_words)
                tt = time.time() - begin_time
                print >>sys.stderr, "BlEU score = %f. Time %d s elapsed. Avg decoding time per sentence %f s" % (bleu_score, tt, tt * 1.0 / len(dev_data))

                if len(valid_history) == 0 or bleu_score > max(valid_history):
                    bad_counter = 0
                    print("Saving the model....")
                    nmt_model.save()
                else:
                    bad_counter += 1
                    if bad_counter >= args.patience:
                        print("Early stop!")
                        exit(0)

                valid_history.append(bleu_score)

            loss = nmt_model.get_encdec_loss(src_batch, tgt_batch)
            loss_value = loss.value()
            total_loss += loss_value * bs
            total_examples += bs

            ppl = np.exp(loss_value * bs / sum([len(s) for s in tgt_batch]))
            print >>sys.stderr, "Epoch=%d, Updates=%d, Loss=%f, Avg. Loss=%f, PPL=%f, Time taken=%d s" % \
                                (epoch+1, updates+1, loss_value, total_loss/total_examples, ppl, time.time()-start_time)
            loss.backward()
            trainer.update()


def test(args, config):
    src_vocab, src_id_to_words, src_train = get_vocab(args.train_src, args.src_vocab_size)
    tgt_vocab, tgt_id_to_words, tgt_train = get_vocab(args.train_tgt, args.tgt_vocab_size)
    
    config["src_voc_size"] = len(src_vocab)
    config["tgt_voc_size"] = len(tgt_vocab)
    src_test = get_data(args.test_src, src_vocab)
    tgt_test = get_data(args.test_tgt, tgt_vocab)
    test_data = zip(src_test, tgt_test)

    nmt_model = EncoderDecoder(config)
    nmt_model.load()
    bleu_score, translations = translate(nmt_model, test_data, src_id_to_words, tgt_id_to_words)

    print >>sys.stderr, "BLEU on test data = ", bleu_score
    with open("../obj/" + args.model_name + "_test_hyps.txt", "w") as fout:
        for hyp in translations:
            fout.write(" ".join(hyp) + '\n')


def translate(model, data_pair, src_id_to_words, tgt_id_to_words):
    translations = []
    references = []
    empty = True
    i = 0
    for src_sent, tgt_sent in data_pair:
        scores, samples = model.gen_samples(src_sent, 100)
        sample = samples[np.array(scores).argmin()]

        src = get_sent(src_sent, src_id_to_words)
        tgt = get_sent(tgt_sent, tgt_id_to_words)
        hyp = get_sent(sample, tgt_id_to_words)
        
        if len(hyp) > 2:
            empty = False
        translations.append(hyp[1:-1])

        print >>sys.stderr, "########################" * 5
        print >>sys.stderr, "Src sent: ", " ".join(src[1:-1])
        print >>sys.stderr, "Tgt sent: ", " ".join(tgt[1:-1])
        print >>sys.stderr, "Hypothesis: ", " ".join(hyp[1:-1])

        # print references[-1]
        print translations[-1]

    if empty:
        return 0.0, translations
    i = 0
    with open("../en-de/test.en-de.low.en", "r") as fref:
        for rline in fref:
            i += 1
            if i == 10:
                break
            references.append([rline.split()])
    bleu_score = corpus_bleu(references, translations)
    return bleu_score, translations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--att_dim", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--train_tgt', type=str, default="en-de/train.en-de.low.filt.en")
    parser.add_argument('--train_src', type=str, default="en-de/train.en-de.low.filt.de")
    parser.add_argument('--dev_tgt', type=str, default="en-de/valid.en-de.low.en")
    parser.add_argument('--dev_src', type=str, default="en-de/valid.en-de.low.de")
    parser.add_argument('--test_tgt', type=str, default="en-de/test.en-de.low.en")
    parser.add_argument('--test_src', type=str, default="en-de/test.en-de.low.de")
#     parser.add_argument('--src_vocab_size', type=int, default=3000)
#     parser.add_argument('--tgt_vocab_size', type=int, default=2000)
    parser.add_argument('--src_vocab_size', type=int, default=30000)
    parser.add_argument('--tgt_vocab_size', type=int, default=20000)
    parser.add_argument('--valid_freq', type=int, default=2500)
    parser.add_argument('--load_from')
    parser.add_argument('--concat_readout', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--test', action="store_true", default=False)
    
    parser.add_argument('--dynet-mem', default=1000, type=int)
    parser.add_argument('--random_seed', default=792551808, type=int)
    parser.add_argument('--for_loop_att', action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    config = {}
    config["layers"] = args.layers
    config["emb_size"] = args.emb_size
    config["hidden_dim"] = args.hid_dim
    config["att_dim"] = args.att_dim
    config["beam_size"] = args.beam_size
    config["src_voc_size"] = args.src_vocab_size
    config["tgt_voc_size"] = args.tgt_vocab_size
    config["model_name"] = args.model_name
    config["dropout"] = args.dropout
    config["concat_readout"] = args.concat_readout
    config["for_loop_att"] = args.for_loop_att
    if args.test:
        test(args, config)
    else:
        train(args, config)




