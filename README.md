# UCan-Attention
ATTENTION-BASED NEURAL NETWORK FOR ILL-EXPOSED IMAGE CORRECTION

by Lucas Messias, Paulo Drews, Silvia Botelho.

#### Overexposure results

<p float="left">
  <img src="/images/overexposure/input/000003.png" width="255" />
  <img src="/images/overexposure/input/000019.png" width="255" /> 
  <img src="/images/overexposure/input/000028.png" width="255" />
</p>

<p float="left">
  <img src="/images/overexposure/output/new4_ucan000003.png" width="255" />
  <img src="/images/overexposure/output/new4_ucan000019.png" width="255" /> 
  <img src="/images/overexposure/output/new4_ucan000028.png" width="255" />
</p>

#### Undexposure results

<p float="left">
  <img src="/images/undexposure/input/000003.png" width="255" />
  <img src="/images/undexposure/input/000006.png" width="255" /> 
  <img src="/images/undexposure/input/000023.png" width="255" />
</p>

<p float="left">
  <img src="/images/undexposure/output/new4_ucan000003.png" width="255" />
  <img src="/images/undexposure/output/new4_ucan000006.png" width="255" /> 
  <img src="/images/undexposure/output/new4_ucan000023.png" width="255" />
</p>

## Introduction

This repository is build for the proposed method for ill-exposed image correction 

## Requirements

* Python == 3.6
* Keras == 2.9.0 
* Numpy== 1.20.0

## Get started



### Model Overview

The source files are located in `src` folder. The models are described in `model_definition.py`.

The images above are related to macro archiecture, attention network, restorarion nertowrk and the Attention and Context Aggregation Block (ACAB).  

<p float="left">
  <img src="/images/solution/arquitetura.png"/>
</p>

<p float="left">
  <img src="/images/solution/rede_atencao.png" width="255"  title="Van Gogh, Self-portrait."/>
  <img src="/images/solution/rede_restauracao.png" width="255" title="Van Gogh, Self-portrait."/> 
  <img src="/images/solution/BAAC.png" width="255" title="Van Gogh, Self-portrait."/>
</p>



