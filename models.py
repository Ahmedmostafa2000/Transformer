import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Input, Add, Softmax, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

class Transformer():
    def __init__(self, x_shape, y_shape, num_heads, d_ff):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.num_heads = num_heads
        self.d_ff = d_ff
        
    def multihead_attention_encoder(self):
        # Inputs
        inputs = Input(shape=self.x_shape)
        # Attention
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.x_shape[-1])(inputs, inputs, inputs)
        # Residual
        residual_layer = Add()([inputs, attention_layer])
        # Normalization
        normalized_output = LayerNormalization()(residual_layer)

        model = Model(inputs=inputs, outputs=normalized_output)
        return model
    
    def positional_feed_forward(self):
        # x_shape: (sequence_length, d_model)
        inputs = Input(shape=self.x_shape)
        
        dense1 = Dense(self.d_ff, activation='relu')(inputs)
        dense2 = Dense(self.x_shape[-1])(dense1)
        # Residual
        residual_layer = Add()([inputs, dense2])
        # Normalization
        normalized_output = BatchNormalization()(residual_layer)
        
        model = Model(inputs=inputs, outputs=normalized_output)
        return model
    
    def create_causal_mask(self, length):
        # Creates a lower triangular matrix with ones (1s) and zeros (0s)
        mask = np.tril(np.ones((length, length)))
        return tf.constant(mask, dtype=tf.float32)

    def masked_multihead_attention(self):
        # Inputs
        inputs = Input(shape=self.y_shape)
        # Masked Attention
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.y_shape[-1])(
            inputs, inputs, inputs, attention_mask=self.create_causal_mask(self.y_shape[-2])
        )
        # Residual
        residual_layer = Add()([inputs, attention_layer])
        # Normalization
        normalized_output = LayerNormalization()(residual_layer)
        
        model = Model(inputs=inputs, outputs=normalized_output)
        return model
    
    def output_layer(self):
        decoder_outputs = Input(shape=self.y_shape)
        logits = Dense(self.y_shape[-1])(decoder_outputs)
        # Softmax
        probabilities = Softmax(axis=-1)(logits)
        model = Model(inputs=decoder_outputs, outputs=probabilities)
        return model
    
    def encoder(self):
        inputs = Input(shape=self.x_shape)
        attention_encoder = self.multihead_attention_encoder()(inputs)
        feed_forward_encoder = self.positional_feed_forward()(attention_encoder)
        model = Model(inputs=inputs, outputs=feed_forward_encoder)
        return model

    def decoder(self):
        q = Input(shape=self.y_shape)
        k = Input(shape=self.x_shape)
        v = Input(shape=self.x_shape)
        # Decoder
        masked_attention_decoder = self.masked_multihead_attention()(q)
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.y_shape[-1], name="testing_attention")(
            query=masked_attention_decoder,
            key=k,
            value=v
        )
        # Residual
        residual_layer = Add()([masked_attention_decoder, attention_layer])
        # Normalization
        normalized_output = LayerNormalization()(residual_layer)
        
        feed_forward_decoder = self.positional_feed_forward()(normalized_output)
        model = Model(inputs=[q, k, v], outputs=feed_forward_decoder)
        return model

    def transformer(self):
        # Encoder
        inputs_1 = Input(shape=self.x_shape)
        inputs_2 = Input(shape=self.y_shape)
        
        encoder_model = self.encoder()(inputs_1)
        
        # Decoder
        masked_attention_decoder = self.masked_multihead_attention()(inputs_2)
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.y_shape[-1], name="testing_attention")(
            query=masked_attention_decoder,
            key=encoder_model,
            value=encoder_model
        )
        # Residual
        residual_layer = Add()([masked_attention_decoder, attention_layer])
        # Normalization
        normalized_output = LayerNormalization()(residual_layer)
        
        feed_forward_decoder = self.positional_feed_forward()(normalized_output)
        # Output
        outputs = self.output_layer()(feed_forward_decoder)
        
        model = Model(inputs=[inputs_1, inputs_2], outputs=outputs)
        return model
