# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# Code developed by Wang Yao (w.yao2@nuigalway.ie) and Muhammad Ali Farooq (m.farooq3@nuigalway.ie)
# The code will genearte 20 random images of boys/ Girls. To ncrease the number of generated images increase input number.

import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import os

def generate_images(network_pkl, num, truncation_psi=0.5):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi
    for i in range(num):
        print('Generating image %d/%d ...' % (i, num))

        # Generate random latent
        z = np.random.randn(1, *Gs.input_shape[1:])

        # Generate image
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars}) 
        images = Gs.run(z, None, **Gs_kwargs)  

        # Save image
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('results/'+str(i)+'.png'))


def main():
    os.makedirs('results/', exist_ok=True)

    network_pkl = 'networks/boys.pkl'
    generate_num = 20#input()

    generate_images(network_pkl, generate_num)

if __name__ == "__main__":
    main()
