![Travis CI](https://travis-ci.com/krasserm/super-resolution.svg?branch=master)

# Single Image Super-Resolution with SRGAN

A [Tensorflow 2.x](https://www.tensorflow.org/beta) based implementation of

- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGAN).

This is a complete re-write of the old Keras/Tensorflow 1.x based implementation available [here](https://github.com/krasserm/super-resolution/tree/previous).
Some parts are still work in progress but you can already train models as described in the papers via a high-level training 
API. Furthermore, you can also SRGAN context.

A `DIV2K` [data provider](#div2k-dataset) automatically downloads [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
training and validation images of given scale (2, 3, 4 or 8) and downgrade operator ("bicubic", "unknown", "mild" or 
"difficult"). 

**Important:** if you want to evaluate the pre-trained models with a dataset other than DIV2K please read. Then you can work on it using 
the flicker2k dataset [click here.](https://www.kaggle.com/hsankesara/flickr-image-dataset)
## Environment setup

Create a new [conda](https://conda.io) environment with

    conda env create -f environment.yml
(The environment.yml file is specifying the required library for installation and name of the environment)
     
or you can install requirements.txt file to install all the dependencies of the project. 
    
     pip install requirements.txt 

activate the environment with below command - 

    conda activate <environment name> "as specified in environmnt.yml file"

## Getting started 

Examples in this section require following pre-trained weights for running (see also example notebooks):  

### Generating super resolution image using SRGAN 

There is a demo folder present in the directory - The user needs to provide the resized images which are of 2k resolution
and convert it to 200 - 300 in pixels (some images are already provided for testing, just provide the image name in the 
home page and check the results) , then the image quality degrades when it is drawn to original size. Thats when the
SRGAN comes into picture and shows you the GAN generated super resolution image. The generated image by the model is saved
in static folder inside output So .you can check the image saved in the output directory present inside the static directory

Directory structure ... 
       
       -> static 
            
            -> output

                 ->image.png

### Pre-trained weights

- [weights-srgan.tar.gz](https://martin-krasser.de/sisr/weights-srgan.tar.gz) 
    - SRGAN as described in the SRGAN paper: 1.55M parameters, trained with VGG54 content loss.
    
After download, extract them in the root folder of the project with

    tar xvfz weights-<...>.tar.gz
### SRGAN

```python
from model.srgan import generator

model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

lr = load_image('demo/image5.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
```

![result-srgan](docs/images/result-srgan.png)

## DIV2K dataset

For training and validation on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) images, applications should use the 
provided `DIV2K` data loader. It automatically downloads DIV2K images to `.div2k` directory and converts them to a 
different format for faster loading.

Crop size in HR images is 96x96.

## SRGAN

### Generator pre-training

```python
from model.srgan import generator
from train import SrganGeneratorTrainer

# Create a training context for the generator (SRResNet) alone.
pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')

# Pre-train the generator with 1,000,000 steps (100,000 works fine too). 
pre_trainer.train(train_ds, valid_ds.take(10), steps=1000000, evaluate_every=1000)

# Save weights of pre-trained generator (needed for fine-tuning with GAN).
pre_trainer.model.save_weights('weights/srgan/pre_generator.h5')
```

### Generator fine-tuning (GAN)

```python
from model.srgan import generator, discriminator
from train import SrganTrainer

# Create a new generator and init it with pre-trained weights.
gan_generator = generator()
gan_generator.load_weights('weights/srgan/pre_generator.h5')

# Create a training context for the GAN (generator + discriminator).
gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())

# Train the GAN with 200,000 steps.
gan_trainer.train(train_ds, steps=200000)

# Save weights of generator and discriminator.
gan_trainer.generator.save_weights('weights/srgan/gan_generator.h5')
gan_trainer.discriminator.save_weights('weights/srgan/gan_discriminator.h5')
```
