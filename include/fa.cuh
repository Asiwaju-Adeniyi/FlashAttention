struct flashForwardkernelConfig{
    const torch::ScalarType dtype;

    const int d_head;
    const int B_r;
    const int B_c;
    const int nWarps;
}