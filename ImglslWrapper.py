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
import moderngl


def critical_error(text):
    raise Exception(f"ImglslWrapper.py: {text}")


def cook_imglsl(text, type_clues, buffer_desc, compile_time_defs=None, match_lines=True, snapshot_name=None):
    """
    :param text: imGLSL text
    :param type_clues: types of values that are present in the array
    :param buffer_desc: list of lists of variable names, as should be present in the buffers
    :param compile_time_defs: definitions that should be substituted inside the imGLSL text
    :param match_lines: output matches the lines of the input if True, otherwise produces more readable text
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

    for buff_i, desc in enumerate(buffer_desc):
        bf_text += f'layout (std430, binding={buff_i}) buffer _GEN_BUFFER_{buff_i} {"{"}\n'
        for name in desc:
            deftext = '    $_GEN_TYPE$ $_GEN_NAME$$_GEN_ARRDATA$;\n' \
                .replace('$_GEN_TYPE$', 'int' if type_clues[name][-1] == 'i' else 'double') \
                .replace('$_GEN_NAME$', name)
            if len(type_clues[name]) > 1:
                deftext = deftext.replace('$_GEN_ARRDATA$', f"[{str(numpy.prod(type_clues[name][:-1]))}]")
            else:
                deftext = deftext.replace('$_GEN_ARRDATA$', '')
            bf_text += deftext
        bf_text += '\n};\n'

    bf_text += '\n/*GLSL-get buffer definitions END*/\n' '//_GEN_BUFFERS'
    if match_lines:
        bf_text = bf_text.replace('\n', ' ')
    text = text.replace('//_GEN_BUFFERS', bf_text)

    # Compile time defs
    if compile_time_defs is not None:
        for cpd in compile_time_defs:
            if str.upper(cpd) != cpd or len(cpd) == 0:
                exception(f'For safety of your feet, all compile-time definitions must be upper case. '
                          f'{cpd} is not.')
            # There is a quirk with this regex - does not match at the start or end of file.
            # Probably not worth the hassle to fix.
            cexpr = re.compile(f'(([^A-Za-z])({cpd})([^A-Za-z]))')
            while True:
                match = cexpr.search(text)
                if match is None:
                    break
                substr, before, var, after = match.groups()
                text = text.replace(substr, f'{before}({compile_time_defs[cpd]}){after}')

    # Final replacements
    for f, t in [('float ', 'double '), ('float(', 'double('),
                 ('float32 ', 'float '), ('float32(', 'float('),
                 (BL, '['), (BR, ']')]:
        text = text.replace(f, t)

    if snapshot_name is not None:
        open(f'out{os.sep}{snapshot_name}.glsl', 'w').write(text)
    return text


def obj_type(obj, name=None):
    """
    Describes a supplied object: measures (if any), type
    :param obj: Object to describe
    :param name: Not needed
    :return: an array, the last element containing type, elements before: measures
    """
    if isinstance(obj, str):
        # Somebody passed a description of an object instead of actually passing the object.
        # Allow this (no validation)
        t = obj.split('_')
        for i in range(len(t) - 1):
            t[i] = int(t[i])
        return t
    if isinstance(obj, int) or type(obj) in [numpy.int32, numpy.int64]:
        return ['i']  # Should be 32-bit int
    if isinstance(obj, float) or type(obj) in [numpy.float32, numpy.float64]:
        return ['f']  # Should be 64-bit float (double)
    if isinstance(obj, list) or type(obj) == numpy.ndarray:
        return [len(obj)] + obj_type(obj[0], name)
    if name is not None:
        critical_error(f'Can not determine type of value {name}')
    else:
        critical_error(f'Can not determine type of some value object')


def type_size(type):
    size = 1
    for dimsize in type[:-1]:
        size *= dimsize
    if type[-1] == 'i':
        size *= 4
    elif type[-1] == 'f':
        size *= 8
    else:
        critical_error(f'Wrong type in objsize')
    return size


def assert_type_match(type, obj, name=None):
    """Compares type of object with a type array (see objtype function)"""
    actual_type = obj_type(obj, name)
    if type != actual_type:
        if name is not None:
            critical_error(f"Type mismatch on {name}."
                           f"Correct type: {type[name]}, attempted type: {obj_type(obj, name)}")
        else:
            critical_error(f"Type mismatch of some object."
                           f"Correct type: {type[name]}, attempted type: {obj_type(obj, name)}")


class ImglslWrapper:
    def __init__(self, _ctx):
        self.buffers: list[moderngl.Buffer] = []
        self.types: dict[str, list] = {}
        self.object_mapping: dict[str, (int, int)] = {}  # value_name->(buf_i, offset)
        if type(_ctx) != moderngl.context.Context and _ctx is not False:
            critical_error('You are supposed to supply a ModernGL context')
        self.ctx: moderngl.context.Context = _ctx

    def _init_buffer(self, size):
        if self.ctx is False:
            self.buffers += [False]
            return len(self.buffers) - 1  # This is a test, no actual buffer needed
        _buffer = self.ctx.buffer(reserve=size)
        self.buffers += [_buffer]
        buff_i = len(self.buffers) - 1
        _buffer.bind_to_storage_buffer(buff_i)
        return buff_i

    def cook_imglsl(self, text, compile_time_defs=None, match_lines=True, shader_name=None):
        """Translate imGLSL to GLSL. After translating, you would probably want to actually run the GLSL."""
        buffer_desc = [[] for _ in range(len(self.buffers))]
        for name in self.object_mapping:
            buff_i, offset = self.object_mapping[name]
            buffer_desc[buff_i] += [(offset, name)]
        buffer_desc = [[v[1] for v in sorted(arr, key=lambda kv: kv[0])] for arr in buffer_desc]
        return cook_imglsl(text, self.types, buffer_desc, compile_time_defs, match_lines, snapshot_name=shader_name)

    def set(self, name, obj):
        """Sets a new value for an already defined object."""
        if name not in self.object_mapping:
            critical_error(f"Can not set an object ({name}) that was not yet defined. "
                           f"Use define_set instead.")
        assert_type_match(self.types[name], obj, name)
        buf_i, offset = self.object_mapping[name]
        if len(self.types[name]) == 1:
            obj = [obj]
        dt = '<f8' if self.types[name][-1] == 'f' else '<i4'
        obj = numpy.array(obj, dtype=dt)
        if self.ctx is False:
            return  # This is a test, no actual set needed
        self.buffers[buf_i].write(obj, offset=offset)

    def define_set(self, env):
        """Defines and sets objects (integers, floats, arrays of integers or floats)
        that can be used in a shader - creating no more than one buffer."""
        # Ensuring that object types are stored
        for name in env:
            if name not in self.types:
                self.types[name] = obj_type(env[name], name)
        # Creating mapped arrays
        # OpenGL requires the values in the buffer to be aligned
        # (otherwise head-scratching issues arise)
        # I managed to get away with not understanding how actually it works
        # please do not break it
        offsets = {'i': 0, 'f': 0}
        for name in env:
            if name not in self.object_mapping:
                needed_size = type_size(self.types[name])
                offsets[self.types[name][-1]] += needed_size
        for type in offsets:
            if offsets[type] == 0:
                continue
            buffer_i = self._init_buffer(offsets[type])
            offset = 0
            for name in env:
                if self.types[name][-1] != type:
                    continue
                needed_size = type_size(self.types[name])
                self.object_mapping[name] = (buffer_i, offset)
                offset += needed_size
        # Assigning values
        for name in env:
            self.set(name, env[name])

    def get(self, name):
        """Retrieves a value from the GPU"""
        if self.ctx is False:
            critical_error(f"This ImglslWrapper is actually a text object: it has no buffers to hold data in."
                           f"You did not supply moderngl context to it while initializing.")
        if name not in self.object_mapping:
            critical_error(f'No object with name {name} in the context.')

        buf_i, offset = self.object_mapping[name]
        dt = '<f8' if self.types[name][-1] == 'f' else '<i4'
        ret = numpy.frombuffer(
            self.buffers[buf_i].read(type_size(self.types[name]), offset=offset),
            dtype=dt)
        if len(ret) == 1:
            return ret[0]
        ret.shape = self.types[name][:-1]
        return ret


# For debugging
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
}
"""

    env = {
        'di': 1,
        'ni': numpy.int64(1),
        'df': 1.248,
        'nf': numpy.float32(1.248),
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

    test_wrapper = ImglslWrapper(False)
    test_wrapper.define_set(env)
    test_wrapper.define_set({'new': 1.23})
    textout = test_wrapper.cook_imglsl(textin, match_lines=False)

    print(f"{'='*15} env\n{env}\n\n"
          f"{'='*15} textin\n{textin}\n"
          f"{'='*15} textout\n{textout}\n")
