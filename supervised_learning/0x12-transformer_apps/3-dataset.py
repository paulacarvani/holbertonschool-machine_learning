#!/usr/bin/env python3
"""
Class Dataset that loads and preps a dataset for machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and preps a dataset
    """
    def __init__(self, batch_size, max_len):
        """
        Class constructor
        """
        data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                               split="train",
                               as_supervised=True)
        data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                               split="validation",
                               as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_len=max_len):
            filtered = tf.logical_and(tf.size(x) <= max_len,
                                      tf.size(y) <= max_len)
            return filtered

        # filter training and validation by max_len number of tokens
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_valid = self.data_valid.filter(filter_max_length)

        # increase performance by caching training dataset
        self.data_train = self.data_train.cache()

        # shuffle the training dataset
        data_size = sum(1 for data in self.data_train)
        self.data_train = self.data_train.shuffle(data_size)

        # split training and validation datasets into padded batches
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_valid = self.data_valid.padded_batch(batch_size)

        # increase performance by prefetching training dataset
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

    def tokenize_dataset(self, data):
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        pt_start_idx = self.tokenizer_pt.vocab_size
        pt_end_idx = pt_start_idx + 1
        en_start_idx = self.tokenizer_en.vocab_size
        en_end_idx = en_start_idx + 1
        pt_tokens = [pt_start_idx] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_idx]
        en_tokens = [en_start_idx] + self.tokenizer_en.encode(
            en.numpy()) + [en_end_idx]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        pt_encoded, en_encoded = tf.py_function(func=self.encode,
                                                inp=[pt, en],
                                                Tout=[tf.int64, tf.int64])
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        return pt_encoded, en_encoded
