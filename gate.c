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
    cell b = in_8 ^ in_4;
    cell c = in_2 & in_7;
    cell e = in_4 | in_5;
    cell g = in_7 & in_6;
    cell h = in_2 & in_1;
    cell i = in_0 | ~in_4;
    cell j = in_1 & in_2;
    cell k = in_7 ^ in_5;
    cell l = in_7 ^ in_3;
    cell m = in_8 & in_6;
    cell n = in_6 & in_3;
    cell o = in_6 | ~in_0;
    cell p = in_6 & in_0;
    cell q = in_1 & ~in_7;
    cell r = in_8 | in_2;
    cell s = in_8 & in_6;
    cell t = a | in_1;
    cell u = o | in_2;
    cell v = l & in_0;
    cell w = in_1 & k;
    cell x = in_2 & in_7;
    cell y = in_5 ^ in_1;
    cell z = in_4 | in_2;
    cell aa = in_2 & ~in_1;
    cell ab = in_0 ^ in_5;
    cell ac = in_1 ^ in_2;
    cell ae = in_6 & in_5;
    cell ag = j | in_0;
    cell ah = in_5 & ~in_2;
    cell ai = aa & ~in_7;
    cell aj = in_8 | x;
    cell ak = b | in_1;
    cell al = t & u;
    cell am = in_5 ^ in_3;
    cell an = r & i;
    cell ao = s & in_6;
    cell ap = e | m;
    cell aq = v & in_5;
    cell ar = in_1 | in_7;
    cell as = in_7 | v;
    cell at = ab | in_3;
    cell au = in_3 & ~q;
    cell av = n & in_3;
    cell aw = ar | as;
    cell ax = in_6 | h;
    cell ay = au & av;
    cell az = c & in_1;
    cell ba = ak | al;
    cell bb = y & ae;
    cell bc = ai & ~in_1;
    cell be = g & in_2;
    cell bg = in_7 | in_2;
    cell bh = al | in_2;
    cell bi = in_5 & in_0;
    cell bj = w | in_8;
    cell bk = an | in_7;
    cell bl = ba | bb;
    cell bm = in_3 & in_5;
    cell bn = ah & ~in_2;
    cell bo = in_0 | at;
    cell bp = am & in_2;
    cell bq = z ^ aq;
    cell br = aw | ae;
    cell bs = bk & bl;
    cell bt = ax & bj;
    cell bu = bh ^ in_3;
    cell bv = bi | in_6;
    cell bw = in_6 | ag;
    cell bx = bq | bg;
    cell by = in_5 | in_3;
    cell bz = in_7 ^ in_5;
    cell ca = bw & aj;
    cell cb = ap & ~az;
    cell cc = in_7 & ac;
    cell ce = bv | in_8;
    cell cg = bz & ~in_3;
    cell ch = ay & bn;
    cell ci = bp ^ in_0;
    cell cj = at | in_3;
    cell ck = bl | bg;
    cell cl = bu | in_7;
    cell cm = bx | in_7;
    cell cn = p | in_2;
    cell co = cl | in_2;
    cell cp = cc | in_2;
    cell cq = bc & ~cg;
    cell cr = in_8 & ca;
    cell cs = by & bs;
    cell ct = bt & co;
    cell cu = bo | ao;
    cell cv = ci & in_4;
    cell cw = cm | in_7;
    cell cx = ca & by;
    cell cy = in_4 & ~cq;
    cell cz = cw & ~ch;
    cell ea = cn ^ be;
    cell eb = cp | in_2;
    cell ec = in_6 | aj;
    cell ee = in_1 | aq;
    cell eg = cs | ct;
    cell eh = cj | in_3;
    cell ei = cb | in_3;
    cell ej = eh | in_3;
    cell ek = eb & in_1;
    cell el = co | in_5;
    cell em = in_2 ^ cr;
    cell en = eg | in_3;
    cell eo = in_0 | cx;
    cell ep = in_2 & ce;
    cell eq = cz | in_1;
    cell er = ch & ca;
    cell es = cu & br;
    cell et = in_5 & in_3;
    cell eu = en & ck;
    cell ev = ei | ej;
    cell ew = eq | er;
    cell ex = ac & in_7;
    cell ey = ee & ec;
    cell ez = el | in_1;
    cell ga = bm & et;
    cell gb = et & in_3;
    cell gc = ea | ex;
    cell ge = ew | in_1;
    cell gg = in_7 & ep;
    cell gh = ge | ea;
    cell gi = in_7 & eu;
    cell gj = ev & ~az;
    cell gk = gc | cr;
    cell gl = eo | gb;
    cell gm = ez | ep;
    cell gn = gh | gi;
    cell go = gm & ce;
    cell gp = ek & ey;
    cell gq = gj | in_0;
    cell gr = ce | in_1;
    cell gs = eu & gl;
    cell gt = go | em;
    cell gu = gr | gg;
    cell gv = gk | ga;
    cell gw = gt & gn;
    cell gx = gu | cv;
    cell gy = aq & ca;
    cell gz = gw & gx;
    cell ha = gq ^ gg;
    cell hb = ha ^ gp;
    cell hc = gz | ex;
    cell he = gv | cx;
    cell hg = he ^ es;
    cell hh = hc ^ gs;
    cell hi = hb & ~gy;
    cell hj = hh & hi;
    cell hk = hg & cy;
    cell out = hj | hk;
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
