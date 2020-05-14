+++
date = "2020-05-10T08:15:23-4:00"
draft = true
title = "Predicting Abnormal Heart Sounds (Part 1: Signal Processing)"

+++

## Motivation

Data is rampant in the cardiovascular system. Among the first reports of humans using pulse data to correlate with health came from an Egyptian papyrus dating 1500 BC. Hippocrates followed 1000 years later with (mostly) correct theories around blood flow. Then we take a historical journey through religion and political influence for the next 1000 years to slowly piece together the concepts of circulation and valves while dispelling common beliefs that 'natural spirits' control blood flow. All this to say that having more data doesn't always lead to more useful interpretation, and humans have been studying this subject for a long time. This article will dive into the tools we can use to make heart sounds data cleaner and thus more useful. 

## Background

The integration between mechanical and electrical components of the cardiovascular system makes it one of the most interesting parts of human physiology. Pace regulation originates from a small section of cells with active potential in the right atrium. This sparks mechanical contraction. The four chambers function as holding containers for the blood as we separate oxygenated and deoxygenated blood and timely deliver it to different parts of the body. There are various parts along this path that we know of to collect data. A few are below:

1) Electrocardiogram (electrical signal): We are collecting from the electrical system here. This can give you information about how regularly the heart is beating. 

2) Phonocardiogram (heart sounds): This is what you can garner from a stethoscope. This can tell you about how the mechanical system is functioning. Particularly, it can be useful in telling you how well valves are closing and opening as these are some of the loudest sounds.

3) Echocardiogram (image): The image data from an echo can tell you about how blood is flowing through the chambers and valves of the heart. This can pair well with heart sounds data. If there is indication of , for instance, a leaky valve then you should be able to hear the murmor first. The source can be diagnosed as the valve using an echo.

## References

[Cardiac History](http://www.uwomj.com/wp-content/uploads/2013/06/v77n2.22-24.pdf)
