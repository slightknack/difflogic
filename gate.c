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
    cell a = in_4 | in_5;
    cell b = in_4 | ~in_7;
    cell c = in_4 | in_5;
    cell e = in_5 | in_4;
    cell g = in_4 | in_3;
    cell h = in_0 | in_6;
    cell i = in_8 & ~in_6;
    cell j = in_5 ^ in_1;
    cell k = in_7 | in_5;
    cell l = in_3 | in_4;
    cell m = in_2 & in_6;
    cell n = in_7 | ~in_5;
    cell o = in_8 & in_4;
    cell p = in_2 & ~in_7;
    cell q = in_0 ^ in_1;
    cell r = in_3 & in_6;
    cell s = in_4 | in_8;
    cell t = in_7 & in_0;
    cell u = in_3 & in_4;
    cell v = in_2 & in_7;
    cell w = in_4 & in_8;
    cell x = in_2 & in_3;
    cell y = in_1 | in_7;
    cell z = in_1 ^ in_2;
    cell aa = in_5 & in_3;
    cell ab = in_3 ^ in_4;
    cell ac = in_3 & in_8;
    cell ae = in_7 | in_5;
    cell ag = in_1 | ~b;
    cell ah = a & in_1;
    cell ai = in_4 & ~aa;
    cell aj = z & in_4;
    cell ak = g | in_5;
    cell al = in_0 | ae;
    cell am = h & in_5;
    cell an = in_1 | in_0;
    cell ao = x | y;
    cell ap = r & in_6;
    cell aq = in_6 | in_1;
    cell ar = s | t;
    cell as = in_1 | in_3;
    cell at = o | p;
    cell au = in_7 & l;
    cell av = w | m;
    cell aw = q & in_8;
    cell ax = in_4 | in_2;
    cell ay = n | an;
    cell az = i & in_0;
    cell ba = in_1 | in_8;
    cell bb = ag & in_0;
    cell bc = ab ^ am;
    cell be = in_4 & ~ap;
    cell bg = as & at;
    cell bh = c ^ in_6;
    cell bi = ~(j ^ in_3);
    cell bj = in_6 & in_8;
    cell bk = in_1 ^ u;
    cell bl = ay | ap;
    cell bm = in_3 & in_0;
    cell bn = ah | l;
    cell bo = ai & ~in_1;
    cell bp = bk & al;
    cell bq = v | bg;
    cell br = ba | in_0;
    cell bs = az & ~in_6;
    cell bt = in_2 | bc;
    cell bu = in_2 | aq;
    cell bv = bs & in_4;
    cell bw = aj & ~in_5;
    cell bx = in_3 & ~bp;
    cell by = e & ~bg;
    cell bz = m | au;
    cell ca = in_7 | in_2;
    cell cb = bn | in_0;
    cell cc = in_3 & ~bo;
    cell ce = ak | aw;
    cell cg = in_0 | in_8;
    cell ch = k | ac;
    cell ci = bq | am;
    cell cj = ci & ar;
    cell ck = ao & ci;
    cell cl = ch | in_5;
    cell cm = ce | in_3;
    cell cn = av | bg;
    cell co = bz & ca;
    cell cp = cb & in_8;
    cell cq = in_2 | bb;
    cell cr = in_8 | in_0;
    cell cs = ca & by;
    cell ct = bi | co;
    cell cu = co | bw;
    cell cv = cg | in_8;
    cell cw = ae | bj;
    cell cx = cp & cq;
    cell cy = cm | in_5;
    cell cz = u & bu;
    cell ea = bo & ~bm;
    cell eb = in_7 & bt;
    cell ec = in_1 | ct;
    cell ee = in_6 | cs;
    cell eg = cr & in_6;
    cell eh = cu & in_6;
    cell ei = bh | cw;
    cell ej = bx | in_8;
    cell ek = eh | cc;
    cell el = cn | eb;
    cell em = ea | ~cw;
    cell en = eg | bv;
    cell eo = in_0 & ej;
    cell ep = in_5 & el;
    cell eq = cl | in_2;
    cell er = em & ~en;
    cell es = cw | in_6;
    cell et = ec & bl;
    cell eu = be | ~ec;
    cell ev = cx | cc;
    cell ew = ei | es;
    cell ex = br & ~eu;
    cell ey = ev | ee;
    cell ez = et & ek;
    cell ga = ew | in_3;
    cell gb = cj | bv;
    cell gc = ga | in_8;
    cell ge = eq & ey;
    cell gg = ck & ~er;
    cell gh = ee & cy;
    cell gi = ax & al;
    cell gj = cv & gc;
    cell gk = ez | cv;
    cell gl = gh & gk;
    cell gm = gk | in_8;
    cell gn = gm & gg;
    cell go = gi | in_2;
    cell gp = gn | ex;
    cell gq = gp | eo;
    cell gr = gb | gl;
    cell gs = go | gj;
    cell gt = gr | ep;
    cell gu = gt | cz;
    cell gv = bp & gs;
    cell gw = gu | gv;
    cell gx = ge & gq;
    cell out = gw ^ gx;
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
