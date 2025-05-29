#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t cell;

cell conway(cell inp[9]) {
    cell in_0 = inp[0];
    cell in_1 = inp[1];
    cell in_2 = inp[2];
    cell in_3 = inp[3];
    cell in_4 = inp[4];
    cell in_5 = inp[5];
    cell in_6 = inp[6];
    cell in_7 = inp[7];
    cell in_8 = inp[8];
    cell a = in_3 | in_2;
    cell b = in_2 | in_1;
    cell c = in_0 & in_5;
    cell e = in_3 | in_7;
    cell g = in_8 | in_5;
    cell h = in_7 ^ in_8;
    cell i = in_4 & ~in_3;
    cell j = in_7 | in_6;
    cell k = in_7 ^ in_3;
    cell l = in_7 & in_6;
    cell m = in_6 & in_3;
    cell n = in_6 & in_3;
    cell o = in_0 | in_1;
    cell p = in_1 ^ in_7;
    cell q = in_4 | in_7;
    cell r = in_7 | in_1;
    cell s = in_8 & in_5;
    cell t = in_3 & in_8;
    cell u = in_3 | in_5;
    cell v = in_5 & in_2;
    cell w = in_8 & in_6;
    cell x = in_0 | in_4;
    cell y = in_8 | in_2;
    cell z = in_6 & ~in_1;
    cell aa = in_5 | in_8;
    cell ab = w & in_7;
    cell ac = in_5 & ~h;
    cell ae = m & in_5;
    cell ag = in_7 | in_1;
    cell ah = x | y;
    cell ai = in_6 | e;
    cell aj = in_2 | in_4;
    cell ak = in_4 & ~in_1;
    cell al = in_0 & in_6;
    cell am = in_1 & in_2;
    cell an = q | in_7;
    cell ao = c & in_5;
    cell ap = o | in_1;
    cell aq = in_4 & ~in_2;
    cell ar = in_2 & ~in_1;
    cell as = in_6 | k;
    cell at = in_4 | t;
    cell au = aa & in_6;
    cell av = in_6 | in_1;
    cell aw = in_6 | in_5;
    cell ax = in_1 & in_2;
    cell ay = in_3 | v;
    cell az = in_7 ^ in_1;
    cell ba = ap | p;
    cell bb = in_1 & ah;
    cell bc = in_1 | in_2;
    cell be = in_0 & in_8;
    cell bg = in_1 | in_2;
    cell bh = in_3 | in_6;
    cell bi = aq | in_0;
    cell bj = in_7 ^ in_5;
    cell bk = in_6 | in_5;
    cell bl = s & in_5;
    cell bm = in_0 | in_8;
    cell bn = in_5 | in_8;
    cell bo = bi & in_8;
    cell bp = in_7 | ay;
    cell bq = ~z;
    cell br = in_0 | in_0;
    cell bs = r ^ in_2;
    cell bt = ba | in_3;
    cell bu = in_6 | in_5;
    cell bv = bc | ab;
    cell bw = ag | be;
    cell bx = ~(in_2 ^ g);
    cell by = at | au;
    cell bz = u | in_5;
    cell ca = in_8 & ~k;
    cell cb = aw & b;
    cell cc = bg & bh;
    cell ce = in_7 ^ i;
    cell cg = ae | in_0;
    cell ch = in_7 & a;
    cell ci = a & az;
    cell cj = in_1 | ~l;
    cell ck = an | in_7;
    cell cl = ax & in_2;
    cell cm = ao & h;
    cell cn = bk & aj;
    cell co = j | in_6;
    cell cp = in_8 & in_2;
    cell cq = in_5 | in_8;
    cell cr = in_0 | cl;
    cell cs = bw | in_1;
    cell ct = cn & in_8;
    cell cu = in_5 & av;
    cell cv = in_7 | in_1;
    cell cw = n & bn;
    cell cx = b | in_7;
    cell cy = in_1 | ac;
    cell cz = in_0 & ~bq;
    cell ea = bj ^ in_3;
    cell eb = bs | t;
    cell ec = in_3 | in_6;
    cell ee = in_8 | ~cz;
    cell eg = ci | in_2;
    cell eh = cs | ct;
    cell ei = in_2 | in_0;
    cell ej = bv & co;
    cell ek = as & in_5;
    cell el = am & in_2;
    cell em = in_6 | in_2;
    cell en = br & al;
    cell eo = bz | cr;
    cell ep = bb & ea;
    cell eq = cw & in_3;
    cell er = bo | cp;
    cell es = ca ^ ak;
    cell et = bl & bt;
    cell eu = cc | eq;
    cell ev = cg & eb;
    cell ew = eo | in_5;
    cell ex = in_0 | in_6;
    cell ey = in_2 & ~ar;
    cell ez = aj | in_3;
    cell ga = in_6 & ee;
    cell gb = by | eg;
    cell gc = ei | in_8;
    cell ge = el & in_2;
    cell gg = cb | in_0;
    cell gh = gc | in_8;
    cell gi = ek & in_5;
    cell gj = in_5 & ai;
    cell gk = in_0 & cx;
    cell gl = cy | in_5;
    cell gm = es | ~cj;
    cell gn = en | eh;
    cell go = er | ep;
    cell gp = in_6 & in_3;
    cell gq = in_1 & in_7;
    cell gr = gn | e;
    cell gs = in_6 & in_3;
    cell gt = eu | ch;
    cell gu = ew | in_0;
    cell gv = in_5 & gh;
    cell gw = in_8 & in_1;
    cell gx = cq | ab;
    cell gy = in_4 & em;
    cell gz = cx | cm;
    cell ha = gs & gt;
    cell hb = ec & in_7;
    cell hc = in_5 | gz;
    cell he = hc & gy;
    cell hg = ha | cl;
    cell hh = in_3 | ey;
    cell hi = in_2 | gm;
    cell hj = in_0 & ez;
    cell hk = hh | gk;
    cell hl = in_3 ^ ex;
    cell hm = gx | hg;
    cell hn = ~(ce ^ in_2);
    cell ho = ej & bp;
    cell hp = gt & in_0;
    cell hq = gw | bu;
    cell hr = hi & ck;
    cell hs = gl | ~hn;
    cell ht = hm | gw;
    cell hu = bx & hl;
    cell hv = ev & cu;
    cell hw = hk & gz;
    cell hx = hq | cv;
    cell hy = hb & in_7;
    cell hz = go | gp;
    cell ia = gu | gg;
    cell ib = hp & in_0;
    cell ic = hx & bm;
    cell ie = ~(hy ^ gu);
    cell ig = hz | ga;
    cell ih = gv & in_5;
    cell ii = ho & ia;
    cell ij = hu | gi;
    cell ik = hr & hs;
    cell il = ic | gq;
    cell im = ij | gp;
    cell in = ih | he;
    cell io = ge & in_2;
    cell ip = gr | in;
    cell iq = hv & co;
    cell ir = gb | il;
    cell is = ie | in_8;
    cell it = iq & in_5;
    cell iu = ig | ik;
    cell iv = ii & is;
    cell iw = ib & ht;
    cell ix = io & im;
    cell iy = hw | in_6;
    cell iz = iu | gj;
    cell ja = ~iy;
    cell jb = ja | ~et;
    cell jc = iw | ~ip;
    cell je = iz | hj;
    cell jg = jb & ~it;
    cell jh = jg & ~ix;
    cell ji = ~(iv ^ je);
    cell jj = jc | ~ir;
    cell jk = jh & ~ji;
    cell out = jk & ~jj;
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
