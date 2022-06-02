"""
Coding GLSL shaders directly is really hard.
For the ease of prototyping and development, this project uses
self-developed GLSL-like intermediate representation (imGLSL).
This file is a wrapper that allows writing imGLSL instead of actual GLSL compute shaders.

It abstracts away buffers, only providing functions to interact with
the objects accessible inside the GLSL.
In the GLSL, simply refer to the objects by the names supplied to set_multiple (or set) function.

Compared to the raw GLSL, imGLSL representation also provides some "syntactic sugar":
* Indirect array access with wrapping: some_arr<-3> = other_arr<10000>
* Pseudo-multidimensional arrays: arr[1][3]
* Length pseudo-function: len1(arr), len(arr)
"""

import os
import re
import numpy
from moderngl.context import Context as modernglContext


def critical_error(text):
    raise Exception(f"ImglslWrapper.py: {text}")


def cook_imglsl(text, type_clues, v_bindings, match_lines=True, snapshot_name=None):
    """
    :param text: imGLSL text
    :param type_clues: types of values that are present in the array
    :param v_bindings: values for array bindings
    :param match_lines: matching lines or readability of the shader
    :param snapshot_name: None for a pure call, string for saving GLSL shader text with this name
    :return: GLSL shader text
    """

    def exception(text):
        raise Exception(f"Cooking imGLSL: {text}")

    # Cooking array accesses
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
        if name not in type_clues:
            exception(f"reference to {name}, it is not defined in the type_clues")
        args = g[2]
        def argparse(args):
            split_by = '\U0001F60E'
            for s0 in [']', '>']:
                for s1 in ['[', '<']:
                    args = args.replace(s0 + s1, s0 + split_by + s1)
            return args.split(split_by)
        args = argparse(args)
        if len(args) != len(type_clues[name]) - 1:
            exception(f"array dimentionality mismatch: {name} is {len(type_clues[name]) - 1}D, not {len(args)}D")
        # We do not witch-hunt excessive brackets, they won't matter after the glsl it compiled
        index = ''
        for i, listr in enumerate(args):
            mode = listr[0]
            listr = listr[1:-1]
            if mode == '<':
                # Speed here is more important than absolute correctness
                # If someone tries to wrap more that than that, they have bigger problems
                loc_len = type_clues[name][i]
                listr = f'(({listr}) + {int(loc_len) << 4}) % {loc_len}'
            loc_offset = numpy.prod(type_clues[name][:-1][i + 1:], dtype=numpy.int32)
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
        if name not in type_clues:
            exception(f"reference to {name}, it is not defined in the type_clues")
        if len(type_clues[name]) == 1:
            exception(f"treating {name} as an array, it is not")
        if not digit < len(type_clues[name]) - 1:
            exception(f"array {name} is only {len(type_clues[name]) - 1}D, "
                      f"tried to get length of {digit}-dimension (0-indexed)'")
        replace_with = f'({type_clues[name][digit]})'
        text = text.replace(substr, replace_with)

    # Generating buffer text
    if '//_GEN_BUFFERS' not in text:
        exception('the text must include //_GEN_BUFFERS statement')
    bf_text = '\n/*GLSL-get buffer definitions START*/\n'
    for value_name in type_clues:
        deftext = '$_GEN_TYPE$ $_GEN_NAME$$_GEN_IS_ARR$;'\
            .replace('$_GEN_TYPE$', 'int' if type_clues[value_name][-1] == 'i' else 'double')\
            .replace('$_GEN_NAME$', value_name)\
            .replace('$_GEN_IS_ARR$', '' if len(type_clues[value_name]) == 1 else '[]')
        bf_text += f'layout (std430, binding={v_bindings[value_name]})'\
                   f' buffer _GEN_BUFFER_{value_name} {"{"}\n'
        bf_text += deftext
        bf_text += '\n};\n'
    bf_text += '\n/*GLSL-get buffer definitions END*/\n' '//_GEN_BUFFERS'
    if match_lines:
        bf_text = bf_text.replace('\n', ' ')
    text = text.replace('//_GEN_BUFFERS', bf_text)

    # Final replacements
    for f, t in [('float ', 'double '), ('float(', 'double('), (BL, '['), (BR, ']')]:
        text = text.replace(f, t)

    if snapshot_name is not None:
        open(f'out{os.sep}{snapshot_name}.imglsl', 'w').write(text)
    return text


def objtype(obj, name=None):
    """
    Describes a supplied object: measures (if any), type
    :param obj: Object to describe
    :param name: Not needed
    :return: an array, the last element containing type, elements before: measures
    """
    if isinstance(obj, str):
        # Somebody passed description of an object instead of actually passing the object.
        # Allow this (no validation)
        return obj.split('_')
    if isinstance(obj, int) or type(obj) == numpy.int32:
        return ['i']
    if isinstance(obj, float) or type(obj) == numpy.float64:
        return ['f']
    if isinstance(obj, list) or type(obj) == numpy.ndarray:
        return [len(obj)] + objtype(obj[0], name)
    if name is not None:
        critical_error(f'Can not determine type of value {name}')
    else:
        critical_error(f'Can not determine type of some value object')


def assert_type_match(type, obj, name=None):
    """Compares type of object with a type array (see objtype function)"""
    actual_type = objtype(obj, name)
    if type != actual_type:
        if name is not None:
            critical_error(f"Type mismatch on {name}."
                           f"Correct type: {type[name]}, attempted type: {objtype(obj,name)}")
        else:
            critical_error(f"Type mismatch of some object."
                           f"Correct type: {type[name]}, attempted type: {objtype(obj,name)}")


class ImglslWrapper:
    def __init__(self, _ctx):
        self.v_types = {}
        self.binding_i = 10
        self.v_bindings = {}
        self.buffers = {}
        if type(_ctx) != modernglContext and _ctx is not None:
            critical_error('You are supposed to supply a ModernGL context')
        self.ctx = _ctx

    def _bind_to_buffer(self, name, obj):
        if name in self.v_bindings:
            return
        self.v_bindings[name] = self.binding_i
        self.binding_i += 1
        if self.ctx is None:
            return  # This is a text, no actual buffer needed
        _buffer = self.ctx.buffer(numpy.array(obj))
        _buffer.bind_to_storage_buffer(self.v_bindings[name])
        self.buffers[name] = _buffer

    def cook_imglsl(self, text, match_lines=True, shader_name=None):
        """Translate imGLSL to GLSL. After translating, you would probably want to actually run the GLSL."""
        return cook_imglsl(text, self.v_types, self.v_bindings, match_lines, snapshot_name=shader_name)

    def set(self, name, obj):
        """Sets value/array that can be used in a shader"""
        if name not in self.v_types:
            self.v_types[name] = objtype(obj, name)
        assert_type_match(self.v_types[name], obj, name)
        if name not in self.v_bindings:
            self._bind_to_buffer(name, obj)
        else:
            self.buffers[name].write(numpy.array(obj))

    def set_multiple(self, env):
        """Sets values/arrays that can be used in a shader"""
        for name in env:
            self.set(name, env[name])

    def get(self, name):
        """Retrieves a value from the GPU"""
        if self.ctx is None:
            critical_error(f"This ImglslWrapper is actually a text object: it has no buffers to hold data in."
                           f"You did not supply moderngl context to in while initializing.")
        if name not in self.buffers:
            critical_error('No object with this name in the context.')

        dt = '<f8' if self.v_types[name][-1] == 'f' else '<i4'
        ret = numpy.frombuffer(self.buffers[name].read(), dtype=dt)
        if len(ret) == 1:
            return ret[0]
        ret.shape = self.v_types[name][:-1]
        return ret


# Test for cook and
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

    test_wrapper = ImglslWrapper(None)
    test_wrapper.set_multiple(env)
    textout = test_wrapper.cook_imglsl(textin)

    print(f"{'='*15} env\n{env}\n\n"
          f"{'='*15} textin\n{textin}\n"
          f"{'='*15} textout\n{textout}\n")
