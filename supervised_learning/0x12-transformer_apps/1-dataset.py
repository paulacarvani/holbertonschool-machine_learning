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
    def __init__(self):
        """
        Class constructor
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = (examples['train'],
                                            examples['validation'])

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

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
