import pydub  # https://github.com/jiaaro/pydub
import numpy
import os
import math
import random
from PIL import Image
import moderngl


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


def create_spectrogram_brrr(samples, fr):
    # Preparing the BRRR machine
    context = moderngl.create_context(standalone=True, require=430)  # Nothing in OpenGL works without a context
    shader_str = open('./hair_shader.glsl', 'r').read()

    # Buffer-binding magic
    binding_val_i = 0
    while 'BINDING_VAL' in shader_str:
        shader_str = shader_str.replace('BINDING_VAL', str(binding_val_i), 1)
        shader_str = shader_str.replace('BUFFER_NAME', 'buffer_i_' + str(binding_val_i), 1)
        binding_val_i += 1
    compute_shader = context.compute_shader(shader_str)  # Create Compute Shader
    bfs = []  # Buffers for the sharder
    def new_buffer(_array, _bf):
        _array = numpy.array(_array)
        _buffer = context.buffer(_array)
        _buffer.bind_to_storage_buffer(len(_bf))
        _bf += [_buffer]

    # Creating buffers, putting values
    samples_n = len(samples)
    new_buffer([fr] + samples, bfs)

    hairs_freq = init_hairs_freq(fr)
    hairs_n = len(hairs_freq)
    new_buffer(hairs_freq, bfs)

    _general_friction_coff = 10.07  # The more, the more aggressive is the friction
    friction = [math.pow(x, 0) for x in hairs_freq]
    friction = [f * _general_friction_coff * math.pi / fr for f in friction]
    friction = [1.0 - pow(0.1, f) for f in friction]
    new_buffer(friction, bfs)

    pull = [pow(math.tau * freq / fr, 2) for freq in hairs_freq]
    new_buffer(pull, bfs)

    cycling_speed_aggregate = numpy.zeros(hairs_n * (5 + math.ceil(fr / hairs_freq[0])), dtype=numpy.float64)
    new_buffer(cycling_speed_aggregate, bfs)

    acc = numpy.zeros(hairs_n * samples_n, dtype=numpy.float64)
    new_buffer(acc, bfs)

    # Running the BRRR machine
    compute_shader.run(group_x=hairs_n)

    # Read the buffer, interpret as floats, return
    acc = numpy.frombuffer(bfs[len(bfs) - 1].read(), dtype='<f8')
    acc.shape = (hairs_n, acc.size // hairs_n)
    # arr = numpy.delete(arr, 0, 1)

    return acc


def random_demo():
    test_music_names = os.listdir(os.getcwd() + os.sep + 'test_music' + os.sep)
    print(test_music_names)
    filename = random.choice(test_music_names)
    filename = os.getcwd() + os.sep + 'test_music' + os.sep + filename
    truncate = (17.0, 20.0)
    return filename, truncate


def get_samples(filename, truncate):
    sound = pydub.AudioSegment.from_file(filename)
    sound = sound.set_channels(1)

    if truncate != (-1, -1):
        sound = sound[truncate[0]*1000:truncate[1]*1000]
        sound.fade_in(20)
        sound.fade_out(20)

    sound.export(f"out/{filename}.fragment.mp3", format="mp3")

    all_samples = numpy.array(sound.get_array_of_samples())
    samples_float = numpy.array(all_samples).T.astype(numpy.float32)
    samples_float = samples_float / sound.max

    return samples_float, sound.frame_rate


def save(output, name):
    output = numpy.copy(output)
    if numpy.isnan(output).any():
        print('ACHTUNG! Output has a nan value!!!')
    output[numpy.isnan(output)] = 0

    output = numpy.clip(output, 1e-20, numpy.inf)
    output = numpy.log(output)
    output = numpy.clip(output, -30.0, numpy.inf)
    output_max = numpy.max(output)
    output_min = numpy.min(output)
    output = numpy.clip(output, output_min, numpy.inf)
    output -= output_min
    output /= output_max - output_min
    output **= 2
    output *= 256
    output = numpy.trunc(output)
    output = output.astype(numpy.int8)

    img = Image.fromarray(output, 'L')
    img.save(f'out/{name}.fragment.png')


def debug_brrr():
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
    out = create_spectrogram_brrr(sine, fr)
    save(out, 'demo')
    exit()


if __name__ == '__main__':
    debug_brrr()

    filename = ''
    truncate = (-1, -1)

    if filename == '':
        filename, truncate = random_demo()

    samples_float, fr = get_samples(filename, truncate)
    spg_raw = create_spectrogram_brrr(samples_float, fr)
    save(spg_raw, filename)
