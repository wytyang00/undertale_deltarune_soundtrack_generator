# Undertale Soundtrack Generator

## Description

Just started! Work in progress!

Sources of the files I will be using for training:
* Undertale: Complete OST - https://musescore.com/user/29625/scores/2075346
* Undertale Piano Medley (not sure if I will use this one) - https://musescore.com/user/3550016/scores/1722571

## To-Do

* Come up with a way to represent the data in the midi files
* Choose which neural network(s) to use for the generator
* Convert the midi files into an usable format and divide them into several examples I can use for training

## Requirements

## Update History

### Oct 7th 2018 (just before my bed time)

Inspired by music generation project videos by [carykh on YouTube](https://www.youtube.com/user/carykh), I decided to do something similar with one of my favorite game soundtrack: Undertale OST.

After creating the repo, I looked for the right midi files and I found 2 great midi files (one of which had the entire soundtrack played with piano and the other being a shorter medley). Both of them have only piano tracks, which simplifies my job, and are long enough (I hope) to train my neural networks on.

Then, I downloaded a free midi player to actually listen and check those midi files and converted them into csv files so that I can use them as training examples later.

In case you want it, here is that [free midi player](http://falcosoft.hu/softwares.html#midiplayer) and [midi <-> csv converter](http://www.fourmilab.ch/webtools/midicsv/).
