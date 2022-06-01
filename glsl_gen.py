"""
Coding GLSL shaders directly is really hard.
For the ease of prototyping and development, this project uses
self-developed GLSL-like intermediate representation (imGLSL).
This file is a imGLSL to actual GLSL compute shader translator.

This file abstracts away buffers, providing only functions to interact with them.

Compared to the raw GLSL, imGLSL representation provides this syntax:
* Indirect access with wrapping: some_arr<-3> = other_arr<10000>
* Pseudo-multidimensional arrays: arr[1][3]
* Length pseudo-function: len1(arr), len(arr)
"""

# HARDCODE STARTS
import os
import sys
import re
import numpy
from moderngl.context import Context as modernglContext


def critical_error(text):
    print(f"ERROR in glsl_gen: {text}", file=sys.stderr)
    exit()


def process_buffers(env, one_liner=True):
    # Oneliner means that the lines in GLSL will strictly match lines in imGLSL.
    # However, this makes buffer definitions unreadable
    v_types = {}
    v_bindings = {}
    bf_text = '\n/*GLSL-get buffer definitions START*/\n'

    binding_i = 10
    for value_name in env:
        v_bindings[value_name] = binding_i
        binding_i += 1
        def objtype(obj):
            if isinstance(obj, str):
                # Somebody passed description of an object instead of actually passing the object
                return obj.split('_')
            if isinstance(obj, int) or type(obj) == numpy.int32:
                return ['i']
            if isinstance(obj, float) or type(obj) == numpy.float64:
                return ['f']
            if isinstance(obj, list) or type(obj) == numpy.ndarray:
                return [len(obj)] + objtype(obj[0])
            critical_error(f'Can not generate from value {value_name}')
        v_types[value_name] = objtype(env[value_name])

        deftext = '$_GEN_TYPE$ $_GEN_NAME$$_GEN_IS_ARR$;'\
            .replace('$_GEN_TYPE$', 'int' if v_types[value_name][-1] == 'i' else 'double')\
            .replace('$_GEN_NAME$', value_name)\
            .replace('$_GEN_IS_ARR$', '' if len(v_types[value_name]) == 1 else '[]')
        bf_text += f'layout (std430, binding={v_bindings[value_name]})'\
                   f' buffer _GEN_BUFFER_{value_name} {"{"}\n'
        bf_text += deftext
        bf_text += '\n};\n'

    bf_text += '\n/*GLSL-get buffer definitions END*/\n' '//_GEN_BUFFERS'

    if one_liner:
        bf_text = bf_text.replace('\n', ' ')

    return bf_text, v_bindings, v_types


def cook_text(text, v_types):
    """
    :param text: imGLSL text
    :param v_types: types of values that are present in the array
    :return: GLSL shader text
    """

    # Cooking accessing arrays
    simple_expr = f"\\[[^\\[\\]]+?\\]"
    wrap_expr = f"\\<[^\\[\\]]+?\\>"
    expr = f'(([A-Za-z\\d_-]+)((({simple_expr})|({wrap_expr}))+))'
    cexpr = re.compile(expr)
    BL = '\uff62'  # Do not translate as array access
    BR = '\uff63'
    while True:
        match = cexpr.search(text)
        if match is None:
            break
        g = match.groups()
        mstr = g[0]
        name = g[1]
        if name not in v_types:
            critical_error(f'imGLSL has a {name} array, there is none in the v_types!')
        args = g[2]
        def argparse(args):
            split_by = '\U0001F60E'
            for s0 in [']', '>']:
                for s1 in ['[', '<']:
                    args = args.replace(s0 + s1, s0 + split_by + s1)
            return args.split(split_by)
        args = argparse(args)
        if len(args) != len(v_types[name]) - 1:
            critical_error(
                f"array dimentionality mismatch: {name} is {len(v_types[name])-1}D, not {len(args)}D"
            )
        # We do not witch-hunt excessive brackets, they won't matter after the glsl it compiled
        index = ''
        for i, listr in enumerate(args):
            mode = listr[0]
            listr = listr[1:-1]
            if mode == '<':
                # Speed here is more important than absolute correctness
                # If someone tries to wrap more that than that, they have bigger problems
                loc_len = v_types[name][i]
                listr = f'(({listr}) + {int(loc_len) << 4}) % {loc_len}'
            loc_offset = numpy.prod(v_types[name][:-1][i+1:], dtype=numpy.int32)
            listr = f'({listr}) * {loc_offset}' if loc_offset != 1 else listr
            index = f'{index} + {listr}' if index != '' else listr
        cooked = f'{name}{BL}{index}{BR}'
        text = text.replace(mstr, cooked)

    # Cooking len(...) pseudo-calls
    cexpr = re.compile('(len(\\d*)\\(([A-Za-z\\d_-]+)\\))')
    while True:
        match = cexpr.search(text)
        if match is None:
            break
        substr, digit, name = match.groups()
        digit = int(digit) if digit != '' else 0
        if len(v_types[name]) == 1:
            critical_error(f'Tried to measure length of {name}, which is not an array')
        if not digit < len(v_types[name]) - 1:
            critical_error(f'Array {name} is only {len(v_types[name]) - 1}D,'
                           f'tried to get length of {digit}-dimension (0-indexed)')
        replace_with = f'({v_types[name][digit]})'
        text = text.replace(substr, replace_with)

    # Final replacements
    for f, t in [('float ', 'double '), ('float(', 'double('), (BL, '['), (BR, ']')]:
        text = text.replace(f, t)

    return text


def generate(imtext, env):
    """
    Creates a GLSL shader, provides functions for interacting with the shader inputs=outputs
    :param text: imGLSL text
    :param env: values that should be bound to the shader
    :return: GLSL shader text, initializer and getter lambdas for interacting with the values
    """
    bf_text, v_bindings, v_types = process_buffers(env)
    text = cook_text(imtext, v_types)
    if '//_GEN_BUFFERS' not in text:
        critical_error('The text must include //_GEN_BUFFERS statement')
    text = text.replace('//_GEN_BUFFERS', bf_text, 1)
    open('out' + os.sep + 'recent.imglsl', 'w').write(text)

    buffers = ['_GEN BUFFER', 'do not touche!', {}]

    def initializer(ctx):
        """Initializes context values in the environment.
        Per generate(...) call, call only once."""
        if type(ctx) != modernglContext:
            critical_error('You are supposed to supply an OpenGL context to this function')
        for name in env:
            _array = numpy.array(env[name])
            if _array.dtype not in [numpy.int32, numpy.float64]:
                critical_error(f'Input {name} has an unsupported element format: {_array.dtype}')
            _buffer = ctx.buffer(_array)
            _buffer.bind_to_storage_buffer(v_bindings[name])
            buffers[2][name] = _buffer

    def getter(name):
        """Retrieves a value from the GPU"""
        dt = '<f8' if v_types[name][-1] == 'f' else '<i4'
        ret = numpy.frombuffer(buffers[2][name].read(), dtype=dt)
        if len(ret) == 1:
            return ret[0]
        ret.shape = v_types[name][:-1]
        return ret

    return text, initializer, getter


# Test
if __name__ == '__main__':
    textin = """
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

//_GEN_BUFFERS

void main() {
    int x = 70;
    int y = 5;
    int daaaf_mixed = daaaf<-x>[2]<70>;
    int nai_simple = nai[1];
    int nai_wrap = nai<x+y>;
    float daaf_simple = daaf[0][1];
    float daaf_wrap = daaf<-1><-1>
    int naaf_mixed = naaf<7>[0];
"""

    env = {
        'di': 1,
        'ni': numpy.int32(1),
        'df': 1.248,
        'nf': numpy.float64(1.248),
        'dai': [1, 2],
        'nai': numpy.array([1, 2, 5, 7]),
        'daf': [0.1, 0.3],
        'naf': numpy.array([1.1, 2.3, 5.8, 13.0]),
        'daai': [[1, 2, 3], [4, 5, 6]],
        'naai': numpy.array([[1, 2, 3], [4, 5, 6]]),
        'daaf': [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
        'naaf': numpy.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
        'daaaf': [[[x for x in range(4)]]*3]*2
    }
    textout, _, _ = generate(str(textin), env)
    print(f"{'='*15} env\n{env}\n"
          f"{'='*15} textin\n{textin}\n"
          f"{'='*15} textout\n{textout}\n")