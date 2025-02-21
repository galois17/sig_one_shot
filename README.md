# One-shot learning for multi-band signals (seaweed)

This is one-shot learning with a Siamese network. While one-shot learning sounds great to classify data for a new domain where you have limited labeled data, you do need quite a bit of data to train a Siamese network.  Fortunately, we often have a good amount of data for the original or source domain. 

```python -m main.train  -e 10 -f ../dataset/data_sampled_20250207_162620.parquet```

![Progress](/assets/animated_plot_small.gif)
