import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Input, Add, Softmax, BatchNormalization, Embedding
from tensorflow.keras.models import Model
import numpy as np

class Transformer():
    def __init__(self, x_shape, y_shape, num_heads = 1, d_ff = 1000, encoder_vocab = 5000, embedding_dim = 300, decoder_vocab = 5000, max_len=50):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.encoder_vocab = encoder_vocab
        self.embedding_dim = embedding_dim
        self.decoder_vocab = decoder_vocab
        self.max_len = max_len
        
    def multihead_attention_encoder(self):
        inputs = Input(shape=(self.max_len, self.embedding_dim), name="encoder_input")
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim, name="encoder_attention")(inputs, inputs, inputs)
        residual_layer = Add(name="encoder_add")([inputs, attention_layer])
        normalized_output = LayerNormalization(name="encoder_layer_norm")(residual_layer)

        model = Model(inputs=inputs, outputs=normalized_output, name="encoder")
        return model
    
    def positional_feed_forward(self):
        inputs = Input(shape=(self.max_len, self.embedding_dim), name="feed_forward_input")
        dense1 = Dense(self.d_ff, activation='relu', name="ff_dense1")(inputs)
        dense2 = Dense(self.embedding_dim, name="ff_dense2")(dense1)
        residual_layer = Add(name="ff_add")([inputs, dense2])
        normalized_output = LayerNormalization(name="ff_batch_norm")(residual_layer)
        
        model = Model(inputs=inputs, outputs=normalized_output, name="feed_forward")
        return model
    
    def create_causal_mask(self, length):
        mask = np.tril(np.ones((length, length)))
        return tf.constant(mask, dtype=tf.float32)

    def masked_multihead_attention(self):
        inputs = Input(shape=(self.max_len,self.embedding_dim), name="masked_attention_input")
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim, name="masked_attention")(
            inputs, inputs, inputs, attention_mask=self.create_causal_mask(self.max_len)
        )
        residual_layer = Add(name="masked_attention_add")([inputs, attention_layer])
        normalized_output = LayerNormalization(name="masked_attention_layer_norm")(residual_layer)
        
        model = Model(inputs=inputs, outputs=normalized_output, name="masked_attention")
        return model
    
    def output_layer(self):
        decoder_outputs = Input(shape=(self.max_len, self.embedding_dim), name="output_input")
        logits = Dense(self.embedding_dim, name="output_dense")(decoder_outputs)
        probabilities = Softmax(axis=-1, name="output_softmax")(logits)
        model = Model(inputs=decoder_outputs, outputs=probabilities, name="output_layer")
        return model
    
    def encoder(self):
        inputs = Input(shape=(self.max_len, self.embedding_dim), name="encoder_input")
        attention_encoder = self.multihead_attention_encoder()(inputs)
        feed_forward_encoder = self.positional_feed_forward()(attention_encoder)
        model = Model(inputs=inputs, outputs=feed_forward_encoder, name="encoder")
        return model

    def decoder(self):
        q = Input(shape=(self.max_len, self.embedding_dim), name="decoder_query")
        k = Input(shape=(self.max_len, self.embedding_dim), name="decoder_key")
        v = Input(shape=(self.max_len, self.embedding_dim), name="decoder_value")
        masked_attention_decoder = self.masked_multihead_attention()(q)
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim, name="decoder_attention")(
            query=masked_attention_decoder,
            key=k,
            value=v
        )
        residual_layer = Add(name="decoder_add")([masked_attention_decoder, attention_layer])
        normalized_output = LayerNormalization(name="decoder_layer_norm")(residual_layer)
        feed_forward_decoder = self.positional_feed_forward()(normalized_output)
        model = Model(inputs=[q, k, v], outputs=feed_forward_decoder, name="decoder")
        return model

    def transformer(self):
        inputs_1 = Input(shape=(self.max_len,), name="transformer_input_1")
        inputs_2 = Input(shape=(self.max_len,), name="transformer_input_2")
        
        embedding_input =  Embedding(input_dim=self.encoder_vocab, output_dim=self.embedding_dim, name="embedding_input")(inputs_1)
        embedding_output = Embedding(input_dim=self.decoder_vocab, output_dim=self.embedding_dim, name="embedding_output")(inputs_2)
        
        encoder_model = self.encoder()(embedding_input)
        masked_attention_decoder = self.masked_multihead_attention()(embedding_output)
        
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim, name="transformer_attention")(
            query=masked_attention_decoder,
            key=encoder_model,
            value=encoder_model
        )
        
        residual_layer = Add(name="transformer_add")([masked_attention_decoder, attention_layer])
        normalized_output = LayerNormalization(name="transformer_layer_norm")(residual_layer)
        
        feed_forward_decoder = self.positional_feed_forward()(normalized_output)
        outputs = self.output_layer()(feed_forward_decoder)
        
        model = Model(inputs=[inputs_1, inputs_2], outputs=outputs, name="transformer_model")
        return model