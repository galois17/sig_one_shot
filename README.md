# One-shot learning for multi-band signals (seaweed)

This is one-shot learning with a Siamese network. While one-shot learning sounds great to classify data for a new domain where you have limited labeled data, you do need quite a bit of data to train a Siamese network.  Fortunately, we often have a good amount of data for the original or source domain. 

```python -m main.train  -e 10 -f ../dataset/data_sampled_20250207_162620.parquet```

![Progress](/assets/animated_plot_small.gif)

Example embeddings that the model will produce:
```
embedding1: [[ 0.23407497  0.11254466  0.20707203  0.03722448  0.13764463
-0.42859793
   0.39810008  0.18463567 -0.27019653  0.3504913  -0.205358    0.31611916
  -0.03195359  0.07288637  0.03342298  0.12096418 -0.17236653 -0.07526764
  -0.05711636  0.12436237 -0.18511012 -0.3543905   0.1153779  -0.1441747
  -0.21086447  0.28927615  0.19468153 -0.11790007  0.05025299 -0.23670542
  -0.1162129  -0.2112182 ]],
embedding2: [[-0.00190187 -0.01436907  0.0475
5883 -0.15065552 -0.06161397 -0.15376185
   0.16055244  0.07131747 -0.09842616 -0.00349931  0.05815304  0.12474258
   0.0340565  -0.05873856 -0.04898193 -0.15285355  0.02543128  0.01146908
  -0.08556616  0.06161552 -0.088339    0.01589086  0.07798572  0.03180868
   0.14388429 -0.03694712  0.22869776 -0.00028521  0.02643663 -0.15548131
   0.2201172  -0.12061716]]
```
