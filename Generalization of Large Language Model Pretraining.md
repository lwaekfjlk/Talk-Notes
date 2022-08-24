# Generalization of Large Language Model Pretraining

by Chenyan Xiong from MSR

### **6 key components of doing pre-training**

1. **Data**

   Pretraining Data Quality is really important!

   Cleaning and filtering are necessary. (simply adding Fasttext classifier can improve quality)

- BERT size pretraining data: Wikipedia + Google Book Corpus ~16GB text
- XLNet and RoBERTa size pretraining data: Wikipedia + xxx ~100-200GB text
- T5 size pretraining data: C4 ~ 745GB text
- Larger high-quality pretraining dataset: ClueWeb22 (Bing’s 34TB high-quality web corpus open sourced for research community)

2. **Pretrained Task**

- Pretraining Task Basics: MLM-style task is simply the most optimal pretrained task

- Dynamic Training

  Using heuristic curriculum learning to improve pretraining

  MLM tasks can have slower training later in the progress

- Application Specific

  Some downstream tasks require centrain-specific information/knowledge/semantic/signal

  In Question Answering, focus more on the named entity, use the **Salient Span Masking** pretraining

  In Dense Retrieval, focus more on full-text sequence embedding, use the **Sequence Contrastive Learning**

3. **Model Architecture**

- Transformer Basics: Easy to train than RNN, powerful and robust

  Decoder models are much easier to train and easy to scale up

  Bidirectional Encoder models are very hard to scale up even up to 10B size

  Enc-Dec: Its training and scaling up property is very close to Encoder models

  The decoders can do encoders’ work, but the performance is much worse due to the pretraining task and single-direction property

- Notable Upgrades

  Make Transformer architecture a little bit more complicated can make pretraining, not robust and wipe out the improvements of architecture modification

  Two notable changes that bring robust benefits:

  1. Positional Embeddings: relative position embeddings have more expressive power
  2. Sparse attention patterns: efficiency gains by design. MLM is a local task anyway, the context length is quite fixed.

- Thoughts

  Benefits from Transformer architecture changes often diminish (especially when pretrained with tons of data and on large models [’ the bitter lesson’])

  Architecture Design seems no longer the most efficient way to bring inductive bias

  - In pretraining, inductive bias may not be as important as efficiency or optimization-friendliness
  - After Pretraining, the model decides the exploration spaces not me (prompt inputs might be more effective)
  - Data and training signals carry more information, the model just consumes them

4. **Optimization**

- Optimization Basics

  SGD has various limitations, momentum carries past batches of information and is more stable

  Adam is still one of the best choices 7 years later

  - Needs a burn-in period for momentum
  - Sometimes **beta** needs slight tuning
  - Simple, elegant, and often works the best

- Reducing the Memory Footprint of Optimizer

  The main cost of Adam: GPU memory usage of optimizer states

  Total Memory Cost: parameter + gradient + 1st order component + 2nd order components

- Stable Optimization

  Critical Component in pretraining

  Tons of divergence. Very painfully, many divergences point only appear in the large-scale model

  Help us to stable the training:

  - Tuning Learning Rate and Scheduler (balancing stability and efficiency)
  - Gradient Norm Clipping (trim out outlier in the scholastic learning)
  - Dynamic layer-wise scaling (smaller starting weights for deeper layers)

  Some initialization can help after 1 month of training

5. **Scaling**

- Parallel Training Basics

  communication between GPU is not the bottleneck

  Previously (without Infinity Band) communication overhead is noticeable

  We can reduce communication frequency with Gradient accumulation

- ZERO Optimizer

  Solve the GPU memory consumption becomes a big bottleneck

  optimizer costs the largest memory (2 [parameter] + 2 [gradient] + K [optimizer parameter])

  ZERO partitioning parameter, gradience, and optimizer size

- Challenges

  no clear mapping of optimal design from the base scale to XXXL scale

  hard to conduct thorough research

  many problems only emerge at large scale

  only there is a successful run, it is easy to speed it up but the first run is the most challenging one

6. **Downstream Usage**

   Familiar fields including:enrich LM (retrieval-augmented) / composition (chaining transformers) / zero(few) shot learning / finetuning

   

**A new Regime**

Differences from the pre-BERT era:

- AI systems now respond to our intervention quite differently
- Different bottlenecks in the full ecosystem, many times the challenge is not on the modeling size
- Different model capacity, ability, and behaviors



**New Directions**

- Improving the efficiency of ML models with a holistic view of the full ML stack

  To change the model to relative position embedding requires modifying the CUDA/Apex code

- Data-Driven AI

  Training data as a new way to convey inductive biases

  the model has the capacity and ability to capture whether what’s in the data is correct or wrong

- Understanding and Theory

  The generalization power of pretraining is mostly observed in NLP instead of CV

  In NLP, with tons of supervised labels finetuning, the benefits of pretraining still exist.

  

**Tips for Continual Pre-training task Design**

- Not trying to twist the pre-trained model checkpoint too much, too much twisting by continual pretraining may cause damage to the original pretrained checkpoint
- If doing continual pretraining for knowledge-intensive tasks, try to design tasks to enhance one specific ability for the models. Many of the trained results of continual pretraining may be duplicated with the original version of pretraining