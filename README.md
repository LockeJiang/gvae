# Generate High Fidelity Images With Generative Variational Autoencoder
Code for the paper `Generate High Fidelity Images With Generative Variational Autoencoder`.

Link to [Paper](https://abhinavsagar.github.io/files/gvae.pdf).

## Data

The dataset can be downloaded from [here](https://www.cancerimagingarchive.net/).

## Network Architecture

### Standard VAE vs our Model

![results](images/img1.png)

### Our network architecture

![results](images/img1.png)

## Algorithm

![results](images/img3.png)

## Usage

`pip install tensorflow-gpu numpy scipy matplotlib tqdm`

`python tools/download_mnist.py`

`python gvae/main/train.py`

`python gvae/main/test.py`

## Experiments

![results](images/img4.png)

![results](images/img5.png)

![results](images/img6.png)

## Results

### Generated images a) MNIST b) Fashion MNIST c) TCIA Pancreas CT

![results](images/img7.png)

### Generated MNIST images a) GAN b) WGAN c) VAE d) GVAE

![results](images/img8.png)

## Citing

If you find this code useful in your research, please consider citing the paper:

```
@article{sagargenerate,
  title={Generate High Fidelity Images With Generative Variational Autoencoder},
  author={Sagar, Abhinav}
}
```

## License

```
MIT License

Copyright (c) 2020 Abhinav Sagar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```









