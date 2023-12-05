# DR-DiffuSE

**Update 12.5:**

We have re-tested the method and uploaded the pre-trained model (c_gen, ddpm, and refiner).

---

In our recent work "DOSE: Diffusion Dropout with Adaptive Prior for Speech Enhancement, NeurIPS, 2023", we revisited why condition collapse happens -- (1) error accumulation (by finite iterative steps (mathematical) and learning errors (sample & true errors)), (2) non-dominent position of condition factor. We find the condition factor does help x_t recover x_0. Please read our recent work for more details.

[paper](https://drive.google.com/file/d/1B0PS-N3m-rdREGd1uK0xU4KhLYDFnxI5/view?pli=1) [code](https://github.com/ICDM-UESTC/DOSE)

---

We also investigate DR-DiffuSE to verify whether some conclusions from DOSE are correct. 

The core difference between current version and previous one is: 

- line 328-334 in src/ddpm_trainer.py -- due to the amplitude of spectrom is high, we should start from c_t/y_t, rather than pure Gaussian noise.

Based on this small modification, we conclude that: 

(1) Our variants, c_gen + two-branch version > c_gen concatenation version > y concatenation version (can be observed from both training losses and inference performance)

(2) Even without explicit guidance, diffusion models (concatenation version, two-branch version) with 6 steps have denoising ability, but their performances are worse than that of the deterministic model.

(3) Error accumulation issue exists (align with the claim in DOSE), 6 steps performs better than 200 steps under the no-guidance scenario.

(4) Using a diffusion model to generate augmented data does improve generalization (a little bit).

---

## Brief
This is the implementation of **DR-DiffuSE** (Revisiting Denoising Diffusion Probabilistic Models for Speech Enhancement: Condition Collapse, Efficiency and Refinement) by **PyTorch**. 

[Paper](/asset/data/DR-DiffuSE.pdf) 

In this work, we elicit generative-based speech enhancement methods (e.g., DDPM-based --) by discussing the problem of poor generalization of existing approaches. Despite the significant performance improvements in other domains like audio speech synthesis and image-to-image translation, the performance of DDPM-based speech enhancement methods is generally lower than that of other generative speech enhancement models. We investigate the following drawbacks: (i) condition collapse problem; (ii) trade-offs between effectiveness and efficiency. 

We give a deep-insight analysis of why condition collapse happens in speech enhancement and propose 3 condition-injecting strategies to ameliorate it. We further design a refinement network to (i) calibrate the output of accelerated DDPM, and (ii) train a more robust/generalizable condition generator -- which is important in Conditional DDPMs.

---

<!-- ## Status
**★★★ Still working in progress ★★★**
Since the NSFC application is concentrated in March, these days are relatively busy... -->

## Environment Requirements
We run the code on a computer with RTX-4090, i7 13700KF, and 128G memory. Install the dependencies via anaconda:

```
# create virtual environment
conda create --name dr-diffuse python=3.8.13

# activate environment
conda activate dr-diffuse

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install speech metrics repo:
# Note: be careful with the repo version, chiefly pesq
pip install https://github.com/ludlows/python-pesq/archive/master.zip
pip install pystoi
pip install librosa

# install utils (we use ``wandb`` for logging and ``rich`` for progress display)
pip install wandb
pip install rich
```

## Basic Architecture

We use a UNet-type model from our previous work [Foster Strengths and Circumvent Weaknesses: a Speech Enhancement Framework with Two-branch Collaborative Learning](https://arxiv.org/pdf/2110.05713.pdf) as the basic architecture for speech enhancement. We give the implementation of our basic model in ``src/model/Base.py``, which will be used as the condition generator and also, the refinement network.


## Training Details

(1) Train Base (condition generator, c_gen) model. Open terminal and run:

> python src/train.py

(2) Train DDPM model. Open terminal and run:

for DiffuSEC:

> python src/train_ddpm.py --model DiffuSEC --wandb

for DiffuSEC + condition generator:

> python src/train_ddpm.py --model DiffuSEC --c_gen --wandb

for DiffuSE + condition generator:

> python src/train_ddpm.py --model DiffuSE --c_gen --wandb

for refiner

> python src/joint_finetune.py --fast_sampling --from_base --wandb 

(3) Inference

> python src/test_ddpm.py --model DiffuSE --fast_sampling --c_gen --c_guidance --refine


## Performance

**Voicebank**

*Base*

{'test_mean_csig': 4.390326794116049, 'test_mean_cbak': 3.5873292531912213, 'test_mean_covl': 3.7635391559208635, 'test_mean_pesq': 3.0882322788238525, 'test_mean_ssnr': 9.794663913082164, 
'test_mean_stoi': 0.9484222835928453}

*DR-DiffuSE*

{'test_mean_csig': 4.376428482730735, 'test_mean_cbak': 3.5494795946836346, 'test_mean_covl': 3.742056515878191, 'test_mean_pesq': 3.063969696465048, 'test_mean_ssnr': 9.38235410664146, 'test_mean_stoi': 0.9491473422523825}

**CHIME-4**

*Base*

{'test_mean_csig': 3.0838875300415776, 'test_mean_cbak': 2.627851942026347, 'test_mean_covl': 2.443433905192998, 'test_mean_pesq': 1.8535393489129615, 'test_mean_ssnr': 5.32511011519281, 
'test_mean_stoi': 0.9197356345337052}

*DR-DiffuSE*

{'test_mean_csig': 3.1085706859198368, 'test_mean_cbak': 2.6527395251714823, 'test_mean_covl': 2.47250678702117, 'test_mean_pesq': 1.881716514988379, 'test_mean_ssnr': 5.3934482878713395, 'test_mean_stoi': 0.9234147534103554}


## Acknowledgments
We would like to thank the authors of previous related projects for generously sharing their code and insights:
- [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://github.com/lmnt-com/diffwave)
- [Foster Strengths and Circumvent Weaknesses: a Speech Enhancement Framework with Two-branch Collaborative Learning](https://github.com/judiebig/Foster-Strengths-and-Circumvent-Weaknesses)
- [Conditional Diffusion Probabilistic Model for Speech Enhancement](https://github.com/neillu23/CDiffuSE)
- [A Study on Speech Enhancement Based on Diffusion Probabilistic Model](https://github.com/neillu23/DiffuSE)
- [Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain](https://arxiv.org/abs/2203.17004)

**Special thanks to [Yen-Ju Lu](https://github.com/neillu23) for his kind help!**

## References
If you find the code useful for your research, please consider citing

```
@inproceedings{tai2023revisiting,
  title={Revisiting denoising diffusion probabilistic models for speech enhancement: Condition collapse, efficiency and refinement},
  author={Tai, Wenxin and Zhou, Fan and Trajcevski, Goce and Zhong, Ting},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={11},
  pages={13627--13635},
  year={2023}
}
```
