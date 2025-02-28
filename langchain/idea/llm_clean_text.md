# Attention Is All You Need

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## 1 Introduction

Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states \( {h}_{t} \) , as a function of the previous hidden state \( {h}_{t - 1} \) and the input for position \( t \). This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks and conditional computation, while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as RNNs and CNNs.

## 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations \( \left( {{x}_{1},\ldots ,{x}_{n}}\right) \) to a sequence of continuous representations \( \mathbf{z} = \left( {{z}_{1},\ldots ,{z}_{n}}\right) \). Given \( \mathbf{z} \), the decoder then generates an output sequence \( \left( {{y}_{1},\ldots ,{y}_{m}}\right) \) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

### 3.1 Encoder and Decoder Stacks

Encoder: The encoder is composed of a stack of \( N = 6 \) identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm \( \left( {x + \text{Sublayer}\left( x\right) }\right) \), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension \( {d}_{\text{model }} = {512} \).

Decoder: The decoder is also composed of a stack of \( N = 6 \) identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position \( i \) can depend only on the known outputs at positions less than \( i \).

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension \( {d}_{k} \), and values of dimension \( {d}_{v} \). We compute the dot products of the query with all keys, divide each by \( \sqrt{{d}_{k}} \), and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix \( Q \). The keys and values are also packed together into matrices \( K \) and \( V \). We compute the matrix of outputs as:

\[\operatorname{Attention}\left( {Q,K,V}\right)  = \operatorname{softmax}\left( \frac{Q{K}^{T}}{\sqrt{{d}_{k}}}\right) V\]

The two most commonly used attention functions are additive attention and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of \( \frac{1}{\sqrt{{d}_{k}}} \). Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

#### 3.2.2 Multi-Head Attention

Instead of performing a single attention function with \( {d}_{\text{model }} \)-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values \( h \) times with different, learned linear projections to \( {d}_{k}, {d}_{k} \) and \( {d}_{v} \) dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding \( {d}_{v} \)-dimensional output values. These are concatenated and once again projected, resulting in the final values.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

\[\operatorname{MultiHead}\left( {Q,K,V}\right)  = \operatorname{Concat}\left( {{\text{ head }}_{1},\ldots ,{\text{ head }}_{\mathrm{h}}}\right) {W}^{O}\]

\[\text{where}{\operatorname{head}}_{\mathrm{i}} = \operatorname{Attention}\left( {Q{W}_{i}^{Q},K{W}_{i}^{K},V{W}_{i}^{V}}\right)\]

### 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

\[\operatorname{FFN}\left( x\right)  = \max \left( {0,x{W}_{1} + {b}_{1}}\right) {W}_{2} + {b}_{2}\]

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is \( {d}_{\text{model }} = {512} \), and the inner-layer has dimensionality \( {d}_{ff} = {2048} \).

### 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension \( {d}_{\text{model }} \). We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation.

### 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension \( {d}_{\text{model }} \) as the embeddings, so that the two can be summed. We use sine and cosine functions of different frequencies:

\[P{E}_{\left( pos,2i\right) } = \sin \left( {{pos}/{10000}^{{2i}/{d}_{\text{model }}}}\right)\]

\[P{E}_{\left( pos,2i + 1\right) } = \cos \left( {{pos}/{10000}^{{2i}/{d}_{\text{model }}}}\right)\]

where pos is the position and \( i \) is the dimension. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions.

## 4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations \( \left( {{x}_{1},\ldots ,{x}_{n}}\right) \) to another sequence of equal length \( \left( {{z}_{1},\ldots ,{z}_{n}}\right) \).

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies. 

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires \( O\left( n\right) \) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length \( n \) is smaller than the representation dimensionality \( d \).

## 6 Results

### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than \( 1/4 \) the training cost of the previous state-of-the-art model.

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature.

### 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013.

### 6.3 English Constituency Parsing

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. 

We trained a 4-layer transformer with \( {d}_{\text{model }} = {1024} \) on the Wall Street Journal (WSJ) portion of the Penn Treebank, about 40K training sentences. 

Our results show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar.

## 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. 

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video.

The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.