# Undertale Soundtrack Generator

## Description

**Sample results shared in the links [here](#Jan-8th-2019)!**

Still a work in progress!

This is my own project where I try to generate original music based on the official soundtracks for some of my favorite video games: Undertale and Deltarune!

I had set up the utility functions for conversions between csv and text files.

Conversions between midi and csv files are done by a 3rd party program: [midi <-> csv converter](http://www.fourmilab.ch/webtools/midicsv/)

Also, I am using this simple [free midi player](http://falcosoft.hu/softwares.html#midiplayer) to listen to the music from midi files.

Several models using LSTM are built and put into training & experimentation.

Models tried or currently being tried:
* Plain 3-Layer LSTM with 300 units
* 3-Layer LSTM with skip connections (Best Loss, but too much overfitting)
* 8-Layer Deep LSTM (Best Result so far)

Originally used training data source:
* Undertale: Complete OST - https://musescore.com/user/29625/scores/2075346

New training data sources are organized in source/source_midi/sources.txt

## To-Do

* Put individual soundtrack data together into a single training dataset.

## Requirements and Dependancies

* Python 3.\*
* PyTorch
* Pandas (For [MICI-csv <-> Text] conversion)
* Numpy (For [MICI-csv <-> Text] conversion and visualization)
* Matplotlib (For visualization)

## Update Logs

---

### Jan 8th 2019

First of all, HAPPY NEW YEAR!

I've been keeping myself busy with PyTorch Udacity scholarship challenge, which started on the day before the last update (Nov 9th 2018). This scholarship challenge's last assignment deadline is tomorrow, and I recently finished it with satisfaction. So, I'm planning to continue with this project quite soon.

One more regarding the scholarship program: there's a Side Project Showcase event going on, and I decided to participate with this project. I'll start posting updates on my Facebook and sharing results and samples until I find a way to make my model accessible for other people. (Though I'm not sure if I'll get myself to do that soon)

Speaking of which, I converted one of my best results to a mp3 audio file, selected several parts that were particularly interesting, and then uploaded them all together. I'm sharing the links here:

* Original full soundtrack (approx. 2 hours and 38 minutes): https://audiomack.com/song/dragonoken/generated

* Samples of my choice: https://audiomack.com/album/dragonoken/robots-are-made-of-metal-magic-and-music

Now that there is a sequel(?) of Undertale, Deltarune, I collected new training data! I originally used a single MIDI file that had all the soundtracks in piano version, but I realized that my model learned to play some rather noisy soundtracks as well (like the sound of an elevator...), which was a bit annoying. Now, I have 1 MIDI file for each soundtrack from both Undertale and Deltarune (rearranged for piano by several people on MuseScore). My next challenge will be to put them all together into a single training set...

Looking forward to trying out several techniques I learned and, hopefully, getting better results!

---

### Nov 10th 2018

First of all, I am so embarassed. I realized that I used a Sigmoid activation before using a Softmax, restricting the expressiveness of the model drastically. The lack of learning was due to this setup. In fact, I would say that it was quite miraculous that my model actually ended up learning something. When I removed the last activation functions altogether and used PyTorch's CrossEntropyLoss, they clearly started to learn. There are some images showing some of the improvements below.

By the way, I made the experimentation notebook much cleaner; now it mostly has only the dataset classes, neural nets, and training & predicting & generating sessions. But, it's still more about me trying out things, so the locations for the files might not match.

Also, in addition to my last models, I created another model that has 8 layers of LSTM with 300 units each. I replaced all the training sessions with this deep model and got pretty impressive results:

* Loss and accuracy
![Deep Model Performance Plot](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/model_outputs/deep_stat.png)
* Some of the predicted notes
![Some Predicted Notes](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/model_outputs/deep_predicted.jpg)
* Some of the generated notes
![Some Generated Notes](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/model_outputs/deep_generate.jpg)

While it did get very good at guessing notes one by one, it had some issues with different rhythms playing together (e.g. slow base part and fast treble part). In an attempt to address this problem, I took a new approach: instead of guessing notes one by one, guessing one entire timestep at a time. I created a new dataset to get k-hot vector representations of the notes instead of one-hot vectors. Then, I used Sigmoid instead of Softmax, Binary Cross Entropy Loss instead of normal Cross Entropy Loss, and a threshold instead of the maximum value. In short, the results weren't bad, but they were still dissapointing.

* Loss and accuracy
![New Model Performance Plot](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/model_outputs/new_stat.png)
* Some of the predicted notes
![Some Predicted Notes](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/model_outputs/new_predicted.jpg)
* Some of the generated notes
![Some Generated Notes](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/model_outputs/new_generated.jpg)

The rhythm issues were somewhat fixed with it, but the noise got quite severe. More than that, generating from skretch was such a pain. I had to manually set the threshold, add some randomness, and adjust the output values in order to get any reasonable result.

So, I went back to my original character by character model and now I am continuing to train it further. I am trying to see if it can overfit...

---

### Oct 30th 2018

I again created a jupyter notebook -- this time, to get used to pytorch, to try converting texts into training datasets, and to experiment with my own LSTM models. I'm uploading the notebook, but remember that I made it just for me to figure things out... the code is very messy and some will not work at all.

I turned each character in the text, including nextlines, to a one-hot vector representing either a note (1-128) or a timestep (0). The target for each character is the next following character. I checked that it works as I intended, then made a data loader for the training.

As for the neural networks, I created 3 classes: one with only 1 layer of LSTM, 129 inputs and 129 outputs; one with adjustable LSTM layers, input and output dimensions; and one with 3 LSTM layers, layer normalizations, and skip connections.

The first network, the one with only 1 hidden LSTM layer, was not used for the actual training session, but instead, used to check whether the way I made models was correct.

The second one was the one I used a lot. I created 2 instances of it with different hyperparameters; one had 1 LSTM layer with 300 neurons, the other one had 3 layers with 700. I trained both of them for many hours and the results were... disappointing. They all became extremely biased about the data they were given; they ended up guessing one or few specific notes regardless of the inputs they take. With further training and loss weight tweaking, I managed to get the simpler one give more varied results, but they could not generate any 'music'. The 3-layered model generated 0s only, generating empty music. The 1-layered model often gave me the same results, but sometimes created only one chord; three notes pressed at the start and never released...

The negative log likelyhood losses were about 4.5-4.8 in the early part of the training. This loss went down as low as about 1.3 for the simple model. However, the loss for the model with 3 layers stayed above 4 and kept oscilating.

Later, I modified the training code so that the overall accuracy and the the accuracy excluding timesteps (0s) were reported. It allowed me to observe when the model becomes biased -- that is, when it starts to guess only 0s or any single note. With extra information provided, I could clearly see that the models quickly settled with biased states where some local minimas were located. I tried to adjust the loss for each class and add dropouts -- even though it wasn't overfitting -- but they did not improve.

The third model did give me an interesting result. I fixed the number of LSTM layers, and added skip connections and layer normalizations. At first, this model, too, became biased just like others. However, it suddenly escaped from that state and showed a remarkable increase in accuracies. The loss stayed around 3.8, but the accuracies kept increasing. Excited, I let it train overnight and then checked the prediction for the dataset. It was messy, but clearly imitating the music very closely. Strangely, the loss did not decrease much while the overall accuracy reached almost 90%. I let it generate a music from scratch and got shocked; it, too, generated empty music filled with 0s. After manually feeding it some specific note sequences, I realized the problem here. (I'm not 100% sure about this, but) This model became biased in a 'smart way'; it figured out that the next note is likely be the same note as the current note or just 0. Due to the nature of the method I used for preparing the data, this way of making predictions could achieve a very high accuracy.

As to why it happend only to this model and not to the previous models, my guess is that it's due to the skip connections. I think the skip connections allowed the input data to flow fairly unmodified throughout the network, turning the LSTM layers to approximate-identity mapping functions... Though I'm not sure why it did not happened to one of my simple models.

Next, I'm planning to try 2 things: spliting the data into bigger time interval per timestep, and using an adversarial loss. Hopefully, the first method will reduce the repetitions in the data and the second method will force the model to be coherent.

---

### Oct 21st 2018

Adding a parser or two wasn't really that hard! All the essential functions were already laid out in my notebook, so I didn't have much work to do other than making argument parsers and grabing those functions to my scripts!

I probably still need to add more checks for the arguments, but at least it works.

3 Utility scripts are created in the utils folder. Use them in a terminal (e.g. cmd) and see the help documents!

I'll add more functionalities to visualize.py so that it can show the images, not just saving them.

By the way, I'm considering using another module to read and possibly play midi files. It will allow me to do conversions between midi and csv within my project -- and not using other software that might not work in other platforms -- and to directly play the results!

---

### Oct 20th 2018

There was an issue with my conversion code between midi and text: namely, the loss of data. Even when I tried to account for every single timestep, there was some amount of loss in the number of notes.

This turned out to be due to subsequent notes that are only 1 tick away. Because of the structure I was forcing them in, subsequent notes were converted to one long note when I used them to reconstruct the midi csv file.

However, I came up with a solution to this problem without having to change the whole structures of my data. Instead of having subsequent notes combined into one, I droped every other note in those series in order to seperate those notes and minimize the loss.

The result was quite amazing. With 25 ticks per timestep, the ratio of notes lost went down from 5% to 0.4%. Here are the results I gathered before and after this modification:

![Note Loss Log Screenshot](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/note_losses.jpg)

---

### Oct 19th 2018

It took quite a lot of effort and time for me to understand the components in MIDI files.

I decided to take everything out of them except for the header, end of file, start and end track, and notes; no key signatures, time signatures, tempo change, or any other controls. Moreover, when I turned them into texts, I only took notes and their times.

I started with taking each component out and see how it changes the music. Luckily for me, key signatures and time signatures did not directly impact the sound, so I could just drop them. Some minor controls, such as reverb and sustain, did cause some changes in the music, but they weren't significant so I just dropped them as well.

Tracks were a bit tricky to handle. Notes were separated into 2 tracks. They are played together, but considered as separate parts. To further simplify the format, I removed all Start_tracks and End_tracks other than the ones with lowest time value and highest time value, then changed all track values greater than 1 to 1.

Removing tempo changed the music drastically. Since the pace of a music differed from part to part, having a same tempo throughout the music was... not really good. So, I adjusted all time values according to the tempo changes by stretching and squashing time intervals with different tempos. As a result, even though there is no tempo change, I managed to make the music sound identical to the original one (except I previously removed reverbs and chorus and all).
  
  
Also, I successfully visualized the notes and saved them as jpeg images. Here are the first 3 parts of the music, from 0 sec to approximately 156 sec:

![midi image 1](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/midimage0.jpg)

![midi image 2](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/midimage1.jpg)

![midi image 3](https://github.com/dragonoken/undertale_soundtrack_generator/blob/master/source/images/midimage2.jpg)

After a bit of more testing, I will put the codes into a python script, possibly with argument parsing!

Oh, by the way, I decided not to use the medley. The complete ost midi file is long and good enough for training. I'm planning to augment it later, anyway.

---
### Oct 7th 2018 (just before my bed time)

Inspired by videos by [carykh on YouTube](https://www.youtube.com/user/carykh), I decided to do something similar with one of my favorite game soundtrack: Undertale OST.

After creating the repo, I looked for the right midi files and I found 2 great midi files (one of which had the entire soundtrack played with piano and the other being a shorter medley). Both of them have only piano tracks, which simplifies my job, and are long enough (I hope) to train my neural networks on.

Then, I downloaded a free midi player to actually listen and check those midi files and converted them into csv files so that I can use them as training examples later.

In case you want it, here is that [free midi player](http://falcosoft.hu/softwares.html#midiplayer) and [midi <-> csv converter](http://www.fourmilab.ch/webtools/midicsv/).
