#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t cell;

cell conway(cell in[9]) {
    cell in_0 = in[0];
    cell in_1 = in[1];
    cell in_2 = in[2];
    cell in_3 = in[3];
    cell in_4 = in[4];
    cell in_5 = in[5];
    cell in_6 = in[6];
    cell in_7 = in[7];
    cell in_8 = in[8];
    cell qay = in_0 & in_4;
    cell u = in_8 ^ in_4;
    cell s = in_2 & in_7;
    cell qdg = in_4 | in_5;
    cell qbl = in_7 & in_6;
    cell c = in_2 & in_1;
    cell z = in_0 | ~in_4;
    cell qby = in_1 & in_2;
    cell qbq = in_7 ^ in_5;
    cell qek = in_7 ^ in_3;
    cell qdy = in_8 & in_6;
    cell qcy = in_6 & in_3;
    cell l = in_6 | ~in_0;
    cell qcx = in_6 & in_0;
    cell qdh = in_1 & ~in_7;
    cell qdm = in_8 | in_2;
    cell qdi = in_8 & in_6;
    cell qei = qay | in_1;
    cell d = l | in_2;
    cell qdt = qek & in_0;
    cell qef = in_1 & qbq;
    cell qer = in_2 & in_7;
    cell qcc = in_5 ^ in_1;
    cell qdz = in_4 | in_2;
    cell qdk = in_2 & ~in_1;
    cell qan = in_0 ^ in_5;
    cell qdc = in_1 ^ in_2;
    cell qde = in_6 & in_5;
    cell qdw = qby | in_0;
    cell qeb = in_5 & ~in_2;
    cell b = qdk & ~in_7;
    cell qag = in_8 | qer;
    cell qcg = u | in_1;
    cell o = qei & d;
    cell qda = in_5 ^ in_3;
    cell qbb = qdm & z;
    cell qdv = qdi & in_6;
    cell qac = qdg | qdy;
    cell qcl = qdt & in_5;
    cell qbn = in_1 | in_7;
    cell qcm = in_7 | qdt;
    cell qcq = qan | in_3;
    cell qdu = in_3 & ~qdh;
    cell qcf = qcy & in_3;
    cell a = qbn | qcm;
    cell qee = in_6 | c;
    cell qaz = qdu & qcf;
    cell qbj = s & in_1;
    cell qae = qcg | o;
    cell qep = qcc & qde;
    cell qao = b & ~in_1;
    cell qbh = qbl & in_2;
    cell qax = in_7 | in_2;
    cell qbm = o | in_2;
    cell qbk = in_5 & in_0;
    cell qbu = qef | in_8;
    cell qba = qbb | in_7;
    cell qcn = qae | qep;
    cell qap = in_3 & in_5;
    cell p = qeb & ~in_2;
    cell qam = in_0 | qcq;
    cell qbd = qda & in_2;
    cell qab = qdz ^ qcl;
    cell qbw = a | qde;
    cell r = qba & qcn;
    cell qad = qee & qbu;
    cell qbs = qbm ^ in_3;
    cell qes = qbk | in_6;
    cell qbg = in_6 | qdw;
    cell qcp = qab | qax;
    cell t = in_5 | in_3;
    cell qcd = in_7 ^ in_5;
    cell qdl = qbg & qag;
    cell qcu = qac & ~qbj;
    cell qas = in_7 & qdc;
    cell y = qes | in_8;
    cell qea = qcd & ~in_3;
    cell qbr = qaz & p;
    cell qar = qbd ^ in_0;
    cell qbz = qcq | in_3;
    cell f = qcn | qax;
    cell qed = qbs | in_7;
    cell qds = qcp | in_7;
    cell i = qcx | in_2;
    cell qbe = qed | in_2;
    cell g = qas | in_2;
    cell qbo = qao & ~qea;
    cell qeo = in_8 & qdl;
    cell n = t & r;
    cell qal = qad & qbe;
    cell qca = qam | qdv;
    cell qdn = qar & in_4;
    cell qdo = qds | in_7;
    cell qeq = qdl & t;
    cell x = in_4 & ~qbo;
    cell qch = qdo & ~qbr;
    cell qcv = i ^ qbh;
    cell qcb = g | in_2;
    cell v = in_6 | qag;
    cell qcr = in_1 | qcl;
    cell qau = n | qal;
    cell qah = qbz | in_3;
    cell qcs = qcu | in_3;
    cell qcj = qah | in_3;
    cell qeg = qcb & in_1;
    cell qdf = qbe | in_5;
    cell qaa = in_2 ^ qeo;
    cell qdq = qau | in_3;
    cell qdb = in_0 | qeq;
    cell h = in_2 & y;
    cell qec = qch | in_1;
    cell qav = qbr & qdl;
    cell qaf = qca & qbw;
    cell qev = in_5 & in_3;
    cell qbf = qdq & f;
    cell qej = qcs | qcj;
    cell qdx = qec | qav;
    cell qel = qdc & in_7;
    cell qdd = qcr & v;
    cell qex = qdf | in_1;
    cell w = qap & qev;
    cell qce = qev & in_3;
    cell qbc = qcv | qel;
    cell qaw = qdx | in_1;
    cell qem = in_7 & h;
    cell qct = qaw | qcv;
    cell qew = in_7 & qbf;
    cell qaj = qej & ~qbj;
    cell qdr = qbc | qeo;
    cell j = qdb | qce;
    cell qeu = qex | h;
    cell qbx = qct | qew;
    cell qaq = qeu & y;
    cell qcw = qeg & qdd;
    cell k = qaj | in_0;
    cell qen = y | in_1;
    cell q = qbf & j;
    cell qdp = qaq | qaa;
    cell qai = qen | qem;
    cell m = qdr | w;
    cell qdj = qdp & qbx;
    cell qck = qai | qdn;
    cell qcz = qcl & qdl;
    cell qet = qdj & qck;
    cell qbv = k ^ qem;
    cell qco = qbv ^ qcw;
    cell qat = qet | qel;
    cell qbp = m | qeq;
    cell qeh = qbp ^ qaf;
    cell qbt = qat ^ q;
    cell qbi = qco & ~qcz;
    cell e = qbt & qbi;
    cell qak = qeh & x;
    cell out = e | qak;
    return out;
}

typedef struct {
    cell* cells;
    size_t cells_len;
    size_t width;
    size_t height;
} board_t;

void fatal(const char* message) {
    fprintf(stderr, "fatal: %s\n", message);
    exit(1);
}

board_t board_new(size_t width, size_t height) {
    if (width % 64 != 0) { fatal("board width must be multiple of 64"); }

    size_t cells_len = (size_t)(width / 64) * height;
    cell* cells = malloc(cells_len * sizeof(cell));
    if (cells == NULL) { fatal("not able to allocate memory for board "); }

    board_t board = { cells, cells_len, width, height };
    return board;
}

// https://en.wikipedia.org/wiki/Xorshift
// constant is frac(golden_ratio) * 2^64
// global state bad cry me a river
uint64_t rand_state = 0x9e3779b97f4a7c55;

uint64_t rand_uint64_t() {
    uint64_t x = rand_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rand_state = x;
    return x;
}

cell rand_cell() {
    // return 0x0101010101010101;
    return rand_uint64_t();
}

void rand_board_mut(board_t *board) {
    for (size_t i = 0; i < board->cells_len; ++i) {
        board->cells[i] = rand_cell();
    }
}

void board_debug(board_t *board) {
    size_t cells_per_row = board->width / 64;
    for (size_t i = 0; i < board->height; ++i) {
        for (size_t j = 0; j < cells_per_row; ++j) {
            cell current_cell = board->cells[i * cells_per_row + j];
            for (int k = 0; k < 64; k++) {
                if ((current_cell >> k) & 1) {
                    printf("█ ");
                } else {
                    printf("  ");
                }
            }
        }
        printf("|\n");
    }
    for (size_t i = 0; i < board->width * 2; ++i) {
        printf("-");
    }
    printf("\n");
}

void board_step_scratch_mut(
    board_t *board,
    board_t *scratch_left,
    board_t *scratch_right
) {
    size_t cells_per_row = board->width / 64;
    for (size_t i = 0; i < board->height; ++i) {
        cell msb_prev = (board->cells[i * cells_per_row + cells_per_row - 1] >> 63) & 1;
        for (size_t j = 0; j < cells_per_row; ++j) {
            size_t idx = i * cells_per_row + j;
            cell cell_curr = board->cells[idx];
            cell cell_shift = cell_curr << 1;
            cell msb_curr = (cell_curr >> 63) & 1;
            scratch_left->cells[idx] = cell_shift | msb_prev;
            msb_prev = msb_curr;
        }
    }
    for (size_t i = 0; i < board->height; ++i) {
        cell lsb_next = board->cells[i * cells_per_row] & 1;
        for (size_t j = cells_per_row; j-- > 0; ) {
            size_t idx = i * cells_per_row + j;
            cell cell_curr = board->cells[idx];
            cell cell_shift = cell_curr >> 1;
            cell lsb_curr = cell_curr & 1;
            scratch_right->cells[idx] = cell_shift | (lsb_next << 63);
            lsb_next = lsb_curr;
        }
    }
}

void board_step_mut(
    board_t *board,
    board_t *s_left, // scratch
    board_t *s_right,
    board_t *s_out
) {
    board_step_scratch_mut(board, s_left, s_right);

    size_t step = board->width / 64;
    size_t wrap = board->cells_len;

    for (size_t i = 0; i < board->cells_len; i++) {
        cell in[9];
        size_t i_top = (i + wrap - step) % wrap;
        size_t i_bottom = (i + step) % wrap;
        // top row
        in[0] = s_left->cells[i_top];
        in[1] = board->cells[i_top];
        in[2] = s_right->cells[i_top];
        // middle row
        in[3] = s_left->cells[i];
        in[4] = board->cells[i];
        in[5] = s_right->cells[i];
        // bottom row
        in[6] = s_left->cells[i_bottom];
        in[7] = board->cells[i_bottom];
        in[8] = s_right->cells[i_bottom];
        // update output
        s_out->cells[i] = conway(in);
    }

    // double-buffering
    cell* tmp_cells = board->cells;
    board->cells = s_out->cells;
    s_out->cells = tmp_cells;
}

int main() {
    size_t width = 512;
    size_t height = 512;

    board_t board = board_new(width, height);
    board_t sl = board_new(width, height);
    board_t sr = board_new(width, height);
    board_t so = board_new(width, height);

    rand_board_mut(&board);

    for (size_t count = 0; count < 100000; count++) {
        // vvv comment out for benchmarking
        printf("\033[H");
        board_debug(&board);
        printf("Step: %zu\n", count);
        // ^^^ comment out for benchmarking
        board_step_mut(&board, &sl, &sr, &so);
    }

    printf("done!\n");
}
