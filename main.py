import pydub  # https://github.com/jiaaro/pydub
import numpy
import os
import sys
from pathlib import Path
import math
import random
from PIL import Image
import moderngl
from ImglslWrapper import ImglslWrapper
import time


def init_hairs_freq(fr=44100, cf=440, hpo=48, low_cut=20, high_cut=20_000) -> list[float]:
    # Due to the limitation of the shader implementation
    # (somebody's not as good at math), we do not allow high frequencies,
    # as they will make pendulums accelerate to infinity
    high_cut = min(high_cut, int(0.9 * fr / math.tau))

    l_cf = math.log(cf)
    l_hpo = math.log(2) / hpo
    l_low = math.log(low_cut)
    l_high = math.log(high_cut)

    l_cf -= l_low
    l_high -= l_low

    l_high /= l_hpo
    l_cf /= l_hpo

    start = l_cf % 1
    if start > 0.0001:
        start -= 1

    hairs = [start]
    while hairs[-1] < l_high:
        hairs += [hairs[-1] + 1]
    hairs = [math.exp(x * l_hpo + l_low) for x in hairs]

    return hairs


def create_spectrogram(samples, fr,
                       time_per_pixel=10, hpo=48):
    # Creating an OpenGL context
    try:
        ctx = moderngl.create_context(standalone=True, require=430)
    except:
        raise Exception(
            'ERROR! Could not create context. This probably means that the GPU used does not support OpenGL 4.3.\n'
            'If you have more than one GPU (both discrete and integrated count),\n'
            'make sure that you are using the card that supports OpenGL 4.3.\n'
            'If you do not have a GPU that supports OpenGL 4.3, you may still run the algorithm!\n'
            'For this, you need to leverage a driver that uses the CPU instead of the GPU to implement OpenGL;\n'
            'this driver is called llvmpipe.'
        )

    # Preparing the values for the shader run
    env = {}
    ctds = {}

    env['FrameRate'] = fr
    env['Samples'] = samples
    env['HairsFreq'] = init_hairs_freq(fr, hpo=hpo)
    hairs_n = len(env['HairsFreq'])

    _general_friction_coff = 10.07  # The more, the more aggressive is the friction
    friction = 1.0 - pow(0.1, _general_friction_coff * math.pi / fr)
    env['Friction'] = friction

    env['Pull'] = [pow(math.tau * freq / fr, 2) for freq in env['HairsFreq']]

    env['CycAgg'] = numpy.zeros((hairs_n, 5 + math.ceil(fr / env['HairsFreq'][0])), dtype=numpy.float64)

    if time_per_pixel <= 0:  # Lossless mode
        env['Act'] = numpy.zeros((hairs_n, len(samples)), dtype=numpy.float64)
        ctds['IS_LOSSLESS'] = 'true'
    else:
        act_len = int((1 / (time_per_pixel / 1000)) * (len(samples) / fr))
        env['Act'] = numpy.zeros((hairs_n, act_len), dtype=numpy.float64)
        ctds['IS_LOSSLESS'] = 'false'

    env['HairsSpeed'] = numpy.zeros(hairs_n, dtype=numpy.float64)
    env['HairsPos'] = numpy.zeros(hairs_n, dtype=numpy.float64)
    env['ProcessingStart'] = 0
    env['ProcessingEnd'] = 0

    # Preparing the shader to run
    w = ImglslWrapper(ctx)
    w.define_set(env)
    imtext = open('./hair_shader.imglsl', 'r').read()
    glsl_text = w.cook_imglsl(imtext, compile_time_defs=ctds, shader_name='spectrogram')
    hairs_shader = ctx.compute_shader(glsl_text)

    # Running the shader
    start_time = time.time()
    pr_start = 0
    pr_size = 1 << 12
    while True:
        tot = 20
        bl = tot * pr_start // len(samples)
        print(f"\rRunning the shader... {'#'*bl}{'_'*(tot-bl)}. Elapsed: {(time.time() - start_time):.6f}", end='')
        pr_end = min(pr_start + pr_size, len(samples))
        if pr_end == pr_start:
            break
        w.set('ProcessingStart', max(1, pr_start))
        w.set('ProcessingEnd', pr_end)
        hairs_shader.run(group_x=((hairs_n + 63) // 64))
        ctx.finish()
        pr_start = pr_end
    print(f"\rShader run completed! Elapsed: {(time.time() - start_time):.6f}")

    output = w.get('Act')
    return output


def random_demo():
    test_music_names = os.listdir('in')
    if len(test_music_names) == 0:
        print('Sorry, you do not have any demo music in the folder called in...\n'
              'Unfortunately, demo music is not distributed with the code due to copyright restrictions.')
        exit()
    filename = random.choice(test_music_names)
    filename = os.getcwd() + os.sep + 'in' + os.sep + filename
    truncate = (17.0, 20.0)
    return filename, truncate


def get_samples(filename, truncate, outname):
    sound = pydub.AudioSegment.from_file(filename)
    sound = sound.set_channels(1)

    if truncate != (-1, -1):
        if truncate[1] <= 0 or truncate[0] >= sound.duration_seconds:
            raise Exception('Truncation leaves an empty input')
        sound = sound[truncate[0]*1000:truncate[1]*1000]
        sound.fade_in(20)
        sound.fade_out(20)

    sound.export(f"out{os.sep}{outname}.mp3", format="mp3")

    all_samples = numpy.array(sound.get_array_of_samples())
    samples_float = numpy.array(all_samples).T.astype(numpy.float64)
    samples_float = samples_float / sound.max

    return samples_float, sound.frame_rate


def save(output, name):
    output = numpy.copy(output)
    if numpy.isnan(output).any():
        print('ACHTUNG! Output has a nan value!!!')
    output[numpy.isnan(output)] = 0

    output = numpy.clip(output, 1e-8, 1)
    output = numpy.log10(output)
    output += 8.0
    output /= 8.0

    output **= 2.2
    output /= 0.9  # Brighten a little, but sacrifice very loud values
    output = numpy.clip(output, 0.0, 1.0)

    output *= 256
    output = numpy.trunc(output)
    output = output.astype(numpy.int8)

    img = Image.fromarray(output, 'L')
    img.save(f'out/{name}.png')


def debug():
    fr = 44100
    hz = 50
    dur = 0.200

    sine = []
    for i in range(math.floor(fr * dur)):
        coff = i / math.floor(fr * dur)
        coff = math.sin(math.tau * coff - 1/4 * math.tau) + 1
        coff /= 2
        per = math.sin(math.tau * hz * i/fr)
        sine += [coff * per]

    sine += [0.0] * math.floor(fr * 0.200)
    out = create_spectrogram(sine, fr, time_per_pixel=-1)
    save(out, 'debug')
    exit()


if __name__ == '__main__':
    for f in ['in', 'out']:
        Path(f).mkdir(exist_ok=True)
    #debug()

    filename = 'in\\Neofeud - The Arcade.mp3'
    truncate = (0, 100)

    if filename == '':
        filename, truncate = random_demo()

    outname = '.'.join(filename.split(os.sep)[-1].split('.')[:-1])
    print(f'Creating spectrogram for {outname}{"" if truncate == (-1, -1) else f", truncated as {truncate}"}')
    if truncate != (-1, -1):
        outname += f'_{truncate[0]:.3f}_{truncate[1]:.3f}'
    samples_float, fr = get_samples(filename, truncate, outname)

    spg_raw = create_spectrogram(samples_float, fr)
    save(spg_raw, outname)
