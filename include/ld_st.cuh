struct TileLayout{
    const int row_fragments;
    const int col_fragments;
};

struct TensorLDSTConfig{
    const TileLayout GSM;
    const TileLayout RF;

    const bool transposed;
    const int block_size;
    const int smem_cols;

    const int warp_ldst_rows;
    const bool compute_over_entire_block;
};