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
    cell a = in_0 & in_4;
    cell qei = in_8 ^ in_4;
    cell qdz = in_2 & in_7;
    cell qcx = in_4 | in_5;
    cell qds = in_7 & in_6;
    cell r = in_2 & in_1;
    cell qdr = in_0 | ~in_4;
    cell t = in_1 & in_2;
    cell f = in_7 ^ in_5;
    cell qbk = in_7 ^ in_3;
    cell qck = in_8 & in_6;
    cell qbg = in_6 & in_3;
    cell qam = in_6 | ~in_0;
    cell qeo = in_6 & in_0;
    cell qec = in_1 & ~in_7;
    cell qcn = in_8 | in_2;
    cell qdh = in_8 & in_6;
    cell b = a | in_1;
    cell qcd = qam | in_2;
    cell qde = qbk & in_0;
    cell qcy = in_1 & f;
    cell qeg = in_2 & in_7;
    cell qed = in_5 ^ in_1;
    cell qbx = in_4 | in_2;
    cell z = in_2 & ~in_1;
    cell l = in_0 ^ in_5;
    cell qdd = in_1 ^ in_2;
    cell qaf = in_6 & in_5;
    cell qcu = t | in_0;
    cell qex = in_5 & ~in_2;
    cell qep = z & ~in_7;
    cell qdl = in_8 | qeg;
    cell d = qei | in_1;
    cell qbs = b & qcd;
    cell qbh = in_5 ^ in_3;
    cell qcp = qcn & qdr;
    cell qdf = qdh & in_6;
    cell qau = qcx | qck;
    cell p = qde & in_5;
    cell e = in_1 | in_7;
    cell q = in_7 | qde;
    cell qbv = l | in_3;
    cell qek = in_3 & ~qec;
    cell qcw = qbg & in_3;
    cell qeq = e | q;
    cell qcv = in_6 | r;
    cell qev = qek & qcw;
    cell s = qdz & in_1;
    cell qew = d | qbs;
    cell qdk = qed & qaf;
    cell qaq = qep & ~in_1;
    cell g = qds & in_2;
    cell qaw = in_7 | in_2;
    cell qer = qbs | in_2;
    cell qas = in_5 & in_0;
    cell qdn = qcy | in_8;
    cell qap = qcp | in_7;
    cell qcq = qew | qdk;
    cell qbd = in_3 & in_5;
    cell qbn = qex & ~in_2;
    cell qby = in_0 | qbv;
    cell qat = qbh & in_2;
    cell w = qbx ^ p;
    cell qcb = qeq | qaf;
    cell qef = qap & qcq;
    cell qbb = qcv & qdn;
    cell qah = qer ^ in_3;
    cell qbl = qas | in_6;
    cell qay = in_6 | qcu;
    cell qbw = w | qaw;
    cell qbm = in_5 | in_3;
    cell qaz = in_7 ^ in_5;
    cell qdj = qay & qdl;
    cell qao = qau & ~s;
    cell qda = in_7 & qdd;
    cell qee = qbl | in_8;
    cell qem = qaz & ~in_3;
    cell qae = qev & qbn;
    cell qdv = qat ^ in_0;
    cell qag = qbv | in_3;
    cell qdp = qcq | qaw;
    cell qar = qah | in_7;
    cell qel = qbw | in_7;
    cell qbp = qeo | in_2;
    cell qce = qar | in_2;
    cell h = qda | in_2;
    cell qbi = qaq & ~qem;
    cell qcf = in_8 & qdj;
    cell qeu = qbm & qef;
    cell qbz = qbb & qce;
    cell qax = qby | qdf;
    cell qca = qdv & in_4;
    cell u = qel | in_7;
    cell qcc = qdj & qbm;
    cell qac = in_4 & ~qbi;
    cell m = u & ~qae;
    cell qcz = qbp ^ g;
    cell qbu = h | in_2;
    cell qbj = in_6 | qdl;
    cell y = in_1 | p;
    cell qbt = qeu | qbz;
    cell i = qag | in_3;
    cell qdt = qao | in_3;
    cell j = i | in_3;
    cell qej = qbu & in_1;
    cell qaa = qce | in_5;
    cell qal = in_2 ^ qcf;
    cell qen = qbt | in_3;
    cell k = in_0 | qcc;
    cell qaj = in_2 & qee;
    cell qad = m | in_1;
    cell qan = qae & qdj;
    cell qbo = qax & qcb;
    cell qdb = in_5 & in_3;
    cell qeh = qen & qdp;
    cell qdc = qdt | j;
    cell qdm = qad | qan;
    cell qdy = qdd & in_7;
    cell qak = y & qbj;
    cell qdw = qaa | in_1;
    cell o = qbd & qdb;
    cell qea = qdb & in_3;
    cell qcm = qcz | qdy;
    cell qeb = qdm | in_1;
    cell c = in_7 & qaj;
    cell qch = qeb | qcz;
    cell v = in_7 & qeh;
    cell qcg = qdc & ~s;
    cell qcl = qcm | qcf;
    cell qbr = k | qea;
    cell qct = qdw | qaj;
    cell qbf = qch | v;
    cell n = qct & qee;
    cell qav = qej & qak;
    cell qdx = qcg | in_0;
    cell qdo = qee | in_1;
    cell x = qeh & qbr;
    cell qcj = n | qal;
    cell qdi = qdo | c;
    cell qcs = qcl | o;
    cell qet = qcj & qbf;
    cell qdg = qdi | qca;
    cell qdq = p & qdj;
    cell qes = qet & qdg;
    cell qbe = qdx ^ c;
    cell qab = qbe ^ qav;
    cell qci = qes | qdy;
    cell qbc = qcs | qcc;
    cell qai = qbc ^ qbo;
    cell qdu = qci ^ x;
    cell qbq = qab & ~qdq;
    cell qba = qdu & qbq;
    cell qco = qai & qac;
    cell out = qba | qco;
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
        // printf("\033[H");
        // board_debug(&board);
        // printf("Step: %zu\n", count);
        // ^^^ comment out for benchmarking
        board_step_mut(&board, &sl, &sr, &so);
    }

    printf("done!\n");
}
