#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Inputs
layout (std430, binding=BINDING_VAL) buffer BUFFER_NAME
{
    double FRAME_RATE;
    double Samples[]; // Length: samples total
};

layout (std430, binding=BINDING_VAL) buffer BUFFER_NAME
{
    double HairsFreq[]; // Length: hairs total
};

layout (std430, binding=BINDING_VAL) buffer BUFFER_NAME
{
    double Friction[]; // Length: hairs total
};

layout (std430, binding=BINDING_VAL) buffer BUFFER_NAME
{
    double Pull[]; // Length: hairs total
};


// Buffers and Outputs
layout (std430, binding=BINDING_VAL) buffer BUFFER_NAME
{
    double CycAgg[]; // Length: hairs total * some value; wraps
};

layout (std430, binding=BINDING_VAL) buffer BUFFER_NAME
{
    double Act[]; // Length: hairs total * samples total
};


// Returns an index of an hair private part of the array, wrapping if needed
int wi(int i, int len) {
    int hair_i = int(gl_WorkGroupID.x);
    int row = len / HairsFreq.length();
    return hair_i * row + ((i + len) % row);
}

// Main shader code
void main() {
    int hair_i = int(gl_WorkGroupID.x);

    // Bins that constitute one halfsine
    double _halfsine_span = 0.5 * double(FRAME_RATE) / HairsFreq[hair_i];
    highp int _halfsine_span_w = int(_halfsine_span);
    double _halfsine_span_d = _halfsine_span - double(_halfsine_span_w);

    double hair_pos = 0.0;
    double hair_speed = 0.0;
    for (int op_i = 1; op_i < Samples.length(); ++op_i) {
        hair_speed += - Pull[hair_i] * hair_pos; // Acc pull
        hair_speed += (Samples[op_i] - Samples[op_i - 1]); // Acc response
        hair_speed += - hair_speed * Friction[hair_i]; // Acc friction
        hair_pos += hair_speed;

        {
            int cl = CycAgg.length();
            int al = Act.length();

            CycAgg[wi(op_i, cl)] = CycAgg[wi(op_i - 1, cl)] + abs(hair_speed);

            Act[wi(op_i, al)]  = CycAgg[wi(op_i, cl)];
            Act[wi(op_i, al)] -= _halfsine_span_d * CycAgg[wi(op_i - _halfsine_span_w - 1, cl)];
            Act[wi(op_i, al)] -= (- _halfsine_span_d + 1.0) * CycAgg[wi(op_i - _halfsine_span_w, cl)];
            Act[wi(op_i, al)] /= HairsFreq[hair_i];

            //Act[wi(op_i, al)] = Act[wi(op_i - 1, al)];
            //Act[wi(op_i, al)] -= (1.0 - 3.14 * Friction) * Act[wi(op_i - 1, al)];
            //Act[wi(op_i, al)] -= Act[wi(op_i, al)];
        }

        //Act[wi(op_i, Act.length())] = Friction[wi(op_i, Friction.length())];
    }

    for (int op_i = 0; op_i < Samples.length(); ++op_i) {
        int delta = hair_i * Samples.length();

        if (op_i + _halfsine_span_w / 2 < Samples.length()) {
            Act[delta + op_i] = Act[delta + op_i + _halfsine_span_w / 2];
        } else {
            Act[delta + op_i] = 0.0;
        }
    }
}
