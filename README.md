# Undertale Soundtrack Generator

## Description

Still Work in progress!

This is my own project where I try to generate original music using one of my favorite OST, Undertale OST!

I had set up the utility functions for conversions between csv and text files.

Conversions between midi and csv files are done by a 3rd party program: [midi <-> csv converter](http://www.fourmilab.ch/webtools/midicsv/)

Also, I am using this simple [free midi player](http://falcosoft.hu/softwares.html#midiplayer) to listen to the music from midi files.

Actual machine learning model is yet to be built.

Neural Networks to use:
* 3-Layer LSTM with 300 units 

Sources of the file I will be using for training:
* Undertale: Complete OST - https://musescore.com/user/29625/scores/2075346

## To-Do

* Divide converted MIDI texts into several examples I can use for training
* Augment the data by transposing them (This will possibly yield more than 6 times more data)

## Requirements and Dependancies

* Python 3.\*
* PyTorch
* Pandas (For [MICI-csv <-> Text] conversion)
* Numpy (For [MICI-csv <-> Text] conversion and visualization)
* Matplotlib (For visualization)

## Update History

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
