# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:15:29 2018

@author: YQ
"""

import tensorflow as tf


def encoder(X, rnn_dim, z_dim):
    with tf.variable_scope("encoder"):
        cell = tf.nn.rnn_cell.BasicRNNCell(rnn_dim)
        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
        
        mean = tf.layers.dense(states, units=z_dim, activation=None, name="mean")
        log_sigma_z = tf.layers.dense(states, units=z_dim, activation=None, name="log_sigma_z")
        
    return mean, log_sigma_z

def decoder(z, rnn_dim, x_dim, seq_len):
    with tf.variable_scope("decoder"):
        h_dec = [None] * seq_len
        x_out = [None] * seq_len
        logits = [None] * seq_len
        
        cell = tf.nn.rnn_cell.BasicRNNCell(rnn_dim)
        h_dec_0 = tf.layers.dense(z, units=rnn_dim, activation=tf.nn.tanh)
        
        h_dec[0] = h_dec_0
        x_dense = tf.keras.layers.Dense(x_dim)
        for i in range(seq_len):
            logits[i] = x_dense(h_dec[i])
            x_out[i] = tf.nn.sigmoid(logits[i])
            
            if i < (seq_len - 1):
                _, h_dec[i+1] = cell(inputs=x_out[i], state=h_dec[i])
                
        logits = tf.stack(logits, axis=1)
        x_out = tf.stack(x_out, axis=1)
                
    return h_dec_0, logits, x_out
    
def build_loss(x, logits, mean, log_sigma_z):
    # loss = (1/2)sum(1 + log(sigma**2) - mean**2 -sigma**2) + (1/L)sum(log p(x|z))
    with tf.variable_scope("recon_loss"): 
        recon_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x)
        recon_loss = tf.reduce_mean(tf.reduce_sum(recon_loss, axis=1))
        
    with tf.variable_scope("latent_loss"):
        sigma_z = tf.exp(log_sigma_z)
        latent_loss = 0.005 * tf.reduce_sum(
                tf.square(mean) + tf.square(sigma_z) - tf.log(tf.square(sigma_z)+1e-8) - 1, axis=-1)
        latent_loss = tf.reduce_mean(latent_loss)
        
    with tf.variable_scope("total_loss"):
        total_loss = recon_loss + latent_loss
        
    return total_loss, recon_loss, latent_loss

def build_vrae(X, rnn_dim, z_dim):
    _, seq_len, x_dim = X.get_shape().as_list()
    mean, log_sigma_z = encoder(X, rnn_dim, z_dim)
    
    eps = tf.random_normal(tf.shape(mean))
    z = mean + tf.exp(log_sigma_z) * eps
    
    h_dec_0, logits, x_out = decoder(z, rnn_dim, x_dim, seq_len)
    
    total_loss, recon_loss, latent_loss = build_loss(X, logits, mean, log_sigma_z)
    
    outputs = {}
    outputs["h_dec_0"] = h_dec_0
    outputs["z"] = z
    outputs["x_out"] = x_out
    outputs["recon_loss"] = recon_loss
    outputs["latent_loss"] = latent_loss
    outputs["total_loss"] = total_loss
    
    return outputs
            
            
if __name__ == "__main__":
    tf.reset_default_graph()
    
    z = tf.placeholder(tf.float32, (None, 20, 88))
    rnn = build_vrae(z, 500, 20)
