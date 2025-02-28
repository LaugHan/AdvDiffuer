# AdvDiffuer
This is an unofficial implement, where most code are copied from https://github.com/ChicForX/advdiff_impl. I do some changes:
- The Grad-CAM in Paper "AdvDiffuser: Natural Adversarial Example Synthesis with Diffusion Models" is generated from the classifier that we want to attack. But in the origin repo, the author trained another classifier on noise instead of clean data, which leads the mask generated becoming heatmaps of noise. However, we expect the heatmap is graph whose shape is like the correct label, instead of noise, so that we can get more natural information.
- So I mainily changed the `advdiffuser.py` and tried several 'timesteps' to train the DDPM. However, I didn't find a good timesteps that generate high quality images. Maybe we should change the DDPM in `diffusionNet.py` to more rubust model such as classifier guidance DDPM. This is to-do stuff.
- Another detail is that the PGD part of the alg is just a simple gradient descent. The paper uses p-norm to constrain perturbation, but is it necessary for the UAE? This can be verified by change the corresponding code `z_t = z_t + perturbation` in `advdiffuser.py`. I didn't check this, reader interested can try it.

If you have any question or detail needed to be explained, please write issue to this repo to discuss with me! I will reply as soon as possible!.
