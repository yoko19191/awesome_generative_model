Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

# Attention Is All You Need

Ashish Vaswani*

Google Brain

Noam Shazeer*

Google Brain

Niki Parmar*

Google Research

Jakob Uszkoreit*

Google Research

Llion Jones*

Google Research

Aidan N. Gomez \$ {}^{ * } \$

University of Toronto

Łukasz Kaiser*

Google Brain

Illia Polosukhin* †

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

---

*Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

\$ {}^{ \dagger  } \$ Work performed while at Google Brain.

\$ {}^{ \ddagger  } \$ Work performed while at Google Research.

---

## 1 Introduction

Recurrent neural networks, long short-term memory  and gated recurrent  neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation . Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures .

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ${h}_{t}$ ,as a function of the previous hidden state \$ {h}_{t - 1} \$ and the input for position \$ t \$ . This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks  and conditional computation , while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences . In all but a few cases , however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU , ByteNet  and ConvS2S , all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions . In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations .

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks .

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as  and .

## 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure . Here,the encoder maps an input sequence of symbol representations \$ \left$ {{x}_{1},\ldots ,{x}_{n}}\right$ \$ to a sequence of continuous representations \$ \mathbf{z} = \left$ {{z}_{1},\ldots ,{z}_{n}}\right$ \$ . Given \$ \mathbf{z} \$ ,the decoder then generates an output sequence \$ \left$ {{y}_{1},\ldots ,{y}_{m}}\right$ \$ of symbols one element at a time. At each step the model is auto-regressive , consuming the previously generated symbols as additional input when generating the next.

Figure 1: The Transformer - model architecture.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1 respectively.

### 3.1 Encoder and Decoder Stacks

Encoder: The encoder is composed of a stack of \$ N = 6 \$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection  around each of the two sub-layers, followed by layer normalization . That is, the output of each sub-layer is LayerNorm \$ \left$ {x + \text{Sublayer}\left$ x\right$ }\right$ \$ ,where Sublayer$x$is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers,produce outputs of dimension \$ {d}_{\text{model }} = {512} \$ .

Decoder: The decoder is also composed of a stack of \$ N = 6 \$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position \$ i \$ can depend only on the known outputs at positions less than \$ i \$ .

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Figure 2: $left$ Scaled Dot-Product Attention. $right$ Multi-Head Attention consists of several attention layers running in parallel.

#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" $Figure 2$. The input consists of queries and keys of dimension \$ {d}_{k} \$ ,and values of dimension \$ {d}_{v} \$ . We compute the dot products of the query with all keys,divide each by \$ \sqrt{{d}_{k}} \$ ,and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix \$ Q \$ . The keys and values are also packed together into matrices \$ K \$ and \$ V \$ . We compute the matrix of outputs as:

\[\operatorname{Attention}\left$ {Q,K,V}\right$  = \operatorname{softmax}\left$ \frac{Q{K}^{T}}{\sqrt{{d}_{k}}}\right$ V \tag{1}\]

The two most commonly used attention functions are additive attention , and dot-product $multiplicative$ attention. Dot-product attention is identical to our algorithm, except for the scaling factor of \$ \frac{1}{\sqrt{{d}_{k}}} \$ . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of \$ {d}_{k} \$ the two mechanisms perform similarly,additive attention outperforms dot product attention without scaling for larger values of \$ {d}_{k} \$ . We suspect that for large values of \$ {d}_{k} \$ ,the dot products grow large in magnitude,pushing the softmax function into regions where it has extremely small gradients 4 To counteract this effect,we scale the dot products by \$ \frac{1}{\sqrt{{d}_{k}}} \$ .

#### 3.2.2 Multi-Head Attention

Instead of performing a single attention function with \$ {d}_{\text{model }} \$ -dimensional keys,values and queries, we found it beneficial to linearly project the queries,keys and values \$ h \$ times with different,learned linear projections to \$ {d}_{k},{d}_{k} \$ and \$ {d}_{v} \$ dimensions,respectively. On each of these projected versions of queries,keys and values we then perform the attention function in parallel,yielding \$ {d}_{v} \$ -dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

---

\$ {}^{4} \$ To illustrate why the dot products get large,assume that the components of \$ q \$ and \$ k \$ are independent random variables with mean 0 and variance 1 . Then their dot product, \$ q \cdot  k = \mathop{\sum }\limits_{{i = 1}}^{{d}_{k}}{q}_{i}{k}_{i} \$ ,has mean 0 and variance \$ {d}_{k} \$ .

---

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

\[\operatorname{MultiHead}\left$ {Q,K,V}\right$  = \operatorname{Concat}\left$ {{\text{ head }}_{1},\ldots ,{\text{ head }}_{\mathrm{h}}}\right$ {W}^{O}\]

\[\text{where}{\operatorname{head}}_{\mathrm{i}} = \operatorname{Attention}\left$ {Q{W}_{i}^{Q},K{W}_{i}^{K},V{W}_{i}^{V}}\right$ \]

Where the projections are parameter matrices \$ {W}_{i}^{Q} \in  {\mathbb{R}}^{{d}_{\text{model }} \times  {d}_{k}},{W}_{i}^{K} \in  {\mathbb{R}}^{{d}_{\text{model }} \times  {d}_{k}},{W}_{i}^{V} \in  {\mathbb{R}}^{{d}_{\text{model }} \times  {d}_{v}} \$ and \$ {W}^{O} \in  {\mathbb{R}}^{h{d}_{v} \times  {d}_{\text{model }}} \$ .

In this work we employ \$ h = 8 \$ parallel attention layers,or heads. For each of these we use \$ {d}_{k} = {d}_{v} = {d}_{\text{model }}/h = {64} \$ . Due to the reduced dimension of each head,the total computational cost is similar to that of single-head attention with full dimensionality.

#### 3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as .

- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out $setting to \$ - \infty \$ $ all values in the input of the softmax which correspond to illegal connections. See Figure 2

### 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

\[\operatorname{FFN}\left$ x\right$  = \max \left$ {0,x{W}_{1} + {b}_{1}}\right$ {W}_{2} + {b}_{2} \tag{2}\]

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is \$ {d}_{\text{model }} = {512} \$ ,and the inner-layer has dimensionality \$ {d}_{ff} = {2048} \$ .

### 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension \$ {d}_{\text{model }} \$ . We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation,similar to . In the embedding layers,we multiply those weights by \$ \sqrt{{d}_{\text{model }}} \$ .

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. \$ n \$ is the sequence length, \$ d \$ is the representation dimension, \$ k \$ is the kernel size of convolutions and \$ r \$ the size of the neighborhood in restricted self-attention.

Layer TypeComplexity per LayerSequential OperationsMaximum Path LengthSelf-Attention\$ O\left$ {{n}^{2} \cdot  d}\right$ \$\$ O\left$ 1\right$ \$\$ O\left$ 1\right$ \$Recurrent\$ O\left$ {n \cdot  {d}^{2}}\right$ \$\$ O\left$ n\right$ \$\$ O\left$ n\right$ \$Convolutional\$ O\left$ {k \cdot  n \cdot  {d}^{2}}\right$ \$\$ O\left$ 1\right$ \$\$ O\left$ {{\log }_{k}\left$ n\right$ }\right$ \$Self-Attention $restricted$\$ O\left$ {r \cdot  n \cdot  d}\right$ \$\$ O\left$ 1\right$ \$\$ O\left$ {n/r}\right$ \$

### 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension \$ {d}_{\text{model }} \$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed .

In this work, we use sine and cosine functions of different frequencies:

\[P{E}_{\left$ pos,2i\right$ } = \sin \left$ {{pos}/{10000}^{{2i}/{d}_{\text{model }}}}\right$ \]

\[P{E}_{\left$ pos,2i + 1\right$ } = \cos \left$ {{pos}/{10000}^{{2i}/{d}_{\text{model }}}}\right$ \]

where pos is the position and \$ i \$ is the dimension. That is,each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from \$ {2\pi } \$ to \$ {10000} \cdot  {2\pi } \$ . We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions,since for any fixed offset \$ k,P{E}_{{pos} + k} \$ can be represented as a linear function of \$ P{E}_{pos} \$ .

We also experimented with using learned positional embeddings  instead, and found that the two versions produced nearly identical results $see Table 3 row $E$$. We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

## 4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations \$ \left$ {{x}_{1},\ldots ,{x}_{n}}\right$ \$ to another sequence of equal length \$ \left$ {{z}_{1},\ldots ,{z}_{n}}\right$ \$ ,with \$ {x}_{i},{z}_{i} \in  {\mathbb{R}}^{d} \$ ,such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies . Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations,whereas a recurrent layer requires \$ O\left$ n\right$ \$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length \$ n \$ is smaller than the representation dimensionality \$ d \$ ,which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece  and byte-pair  representations. To improve computational performance for tasks involving very long sequences,self-attention could be restricted to considering only a neighborhood of size \$ r \$ in the input sequence centered around the respective output position. This would increase the maximum path length to \$ O\left$ {n/r}\right$ \$ . We plan to investigate this approach further in future work.

A single convolutional layer with kernel width \$ k 

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

ModelBLEUTraining Cost $FLOPs$EN-DEEN-FREN-DEEN-FRByteNet 23.75Deep-Att + PosUnk 39.2\$ {1.0} \cdot  {10}^{20} \$GNMT + RL 24.639.92\$ {2.3} \cdot  {10}^{19} \$\$ {1.4} \cdot  {10}^{20} \$ConvS2S 25.1640.46\$ {9.6} \cdot  {10}^{18} \$\$ {1.5} \cdot  {10}^{20} \$MoE 13226.0340.56\$ {2.0} \cdot  {10}^{19} \$\$ {1.2} \cdot  {10}^{20} \$Deep-Att + PosUnk Ensemble 40.4\$ {8.0} \cdot  {10}^{20} \$GNMT + RL Ensemble 26.3041.16\$ {1.8} \cdot  {10}^{20} \$\$ {1.1} \cdot  {10}^{21} \$ConvS2S Ensemble 26.3641.29\$ {7.7} \cdot  {10}^{19} \$\$ {1.2} \cdot  {10}^{21} \$Transformer $base model$27.338.1\$ {3.3} \cdot  {10}^{18} \$Transformer $big$28.441.8\$ {2.3} \cdot  {10}^{19} \$

Residual Dropout We apply dropout  to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of \$ {P}_{\text{drop }} = {0.1} \$ .

Label Smoothing During training,we employed label smoothing of value \$ {\epsilon }_{ls} = {0.1} \$ . This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

## 6 Results

### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model $Transformer $big$ in Table 2$ outperforms the best previously reported models $including ensembles$ by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models,at less than \$ 1/4 \$ the training cost of the previous state-of-the-art model. The Transformer $big$ model trained for English-to-French used dropout rate \$ {P}_{\text{drop }} = {0.1} \$ ,instead of 0.3 .

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty \$ \alpha  = {0.6} \$ . These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50 , but terminate early when possible .

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU 5

### 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3

---

\$ {}^{5} \$ We used values of 2.8,3.7,6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.

---

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

\$ N \$\$ {d}_{\text{model }} \$\$ {d}_{\mathrm{{ff}}} \$\$ h \$\$ {d}_{k} \$\$ {d}_{v} \$\$ {P}_{\text{drop }} \$\$ {\epsilon }_{ls} \$train stepsPPL $dev$BLEU $dev$params \$ \times  {10}^{6} \$base65122048864640.10.1100K4.9225.865$A$15125125.2924.941281285.0025.51632324.9125.83216165.0125.4$B$165.1625.158325.0125.460$C$26.1123.73645.1925.35084.8825.58025632325.7524.52810241281284.6626.016810245.1225.45340964.7526.290$D$0.05.7724.60.24.9525.50.04.6725.30.25.4725.7$E$positional embedding instead of sinusoids4.9225.7big610244096160.3300K4.3326.4213

In Table 3 rows $A$, we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2 While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

In Table 3 rows $B$,we observe that reducing the attention key size \$ {d}_{k} \$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows $C$ and $D$ that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row $E$ we replace our sinusoidal positional encoding with learned positional embeddings , and observe nearly identical results to the base model.

### 6.3 English Constituency Parsing

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes .

We trained a 4-layer transformer with \$ {d}_{\text{model }} = {1024} \$ on the Wall Street Journal $WSJ$ portion of the Penn Treebank , about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences . We used a vocabulary of \$ {16}\mathrm{\;K} \$ tokens for the WSJ only setting and a vocabulary of \$ {32}\mathrm{\;K} \$ tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual $section 5.4$, learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length +300 . We used a beam size of 21 and \$ \alpha  = {0.3} \$ for both WSJ only and the semi-supervised setting.

Table 4: The Transformer generalizes well to English constituency parsing $Results are on Section 23 of WSJ$

ParserTrainingWSJ 23 F1Vinyals & Kaiser el al. $2014$ WSJ only, discriminative88.3Petrov et al. $2006$ WSJ only, discriminative90.4Zhu et al. $2013$ WSJ only, discriminative90.4Dyer et al. $2016$ WSJ only, discriminative91.7Transformer $4 layers$WSJ only, discriminative91.3Zhu et al. $2013$ semi-supervised91.3Huang & Harper $2009$ semi-supervised91.3McClosky et al. $2006$ semi-supervised92.1Vinyals & Kaiser el al. $2014$ semi-supervised92.1Transformer $4 layers$semi-supervised92.7Luong et al. $2015$ multi-task93.0Dyer et al. $2016$ generative93.3

Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar .

In contrast to RNN sequence-to-sequence models , the Transformer outperforms the Berkeley-Parser  even when training only on the WSJ training set of 40K sentences.

## 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

## References

 Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

 Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

 Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

 Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

 Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

 Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.

 Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

 Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.

 Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

 Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

 Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.

 Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

 Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9$8$:1735-1780, 1997.

 Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832-841. ACL, August 2009.

 Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

 Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, $NIPS$, 2016.

 Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations $ICLR$, 2016.

 Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko-ray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

 Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.

 Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

 Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.

 Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

 Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.

 Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.

 Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19$2$:313–330, 1993.

 David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152-159. ACL, June 2006.

 Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.

 Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

 Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433-440. ACL, July 2006.

 Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.

 Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

 Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

 Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi-nov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15$1$:1929-1958, 2014.

 Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.

 Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104-3112, 2014.

 Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

 Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.

 Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

 Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

 Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51 st Annual Meeting of the ACL $Volume 1: Long Papers$, pages 434-443. ACL, August 2013. Attention Visualizations

Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6 . Many of the attention heads attend to a distant dependency of the verb 'making', completing the phrase 'making...more difficult'. Attentions here shown only for the word 'making'. Different colors represent different heads. Best viewed in color.

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'its' for attention heads 5 and 6 . Note that the attentions are very sharp for this word.

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6 . The heads clearly learned to perform different tasks.

