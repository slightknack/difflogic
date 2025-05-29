#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t cell;

// human attempt, 54 gates
cell conway_clayton(cell in[9]) {
    // pairs
    cell p00 = ~(in[0] | in[1]);
    cell p10 = ~(in[2] | in[3]);
    cell p20 = ~(in[5] | in[6]);
    cell p30 = ~(in[7] | in[8]);
    cell p01 = in[0] ^ in[1];
    cell p11 = in[2] ^ in[3];
    cell p21 = in[5] ^ in[6];
    cell p31 = in[7] ^ in[8];
    cell p02 = in[0] & in[1];
    cell p12 = in[2] & in[3];
    cell p22 = in[5] & in[6];
    cell p32 = in[7] & in[8];
    // halfs
    cell h00 = p00 & p10;
    cell h10 = p20 & p30;
    cell h01 = (p00 & p11) | (p10 & p01);
    cell h11 = (p20 & p31) | (p30 & p21);
    cell h03 = (p02 & p11) | (p12 & p01);
    cell h13 = (p22 & p31) | (p32 & p21);
    cell h02 = (p02 & p10) | (p00 & p12) | (p01 & p11);
    cell h12 = (p22 & p30) | (p20 & p32) | (p21 & p31);
    // neighbors
    cell n2 = (h01 & h11) | (h02 & h10) | (h00 & h12);
    cell n3 = (h03 & h10) | (h13 & h00) | (h02 & h11 ) | (h12 & h01);
    // rule
    cell out = n3 | (in[4] & n2);
    return out;
}

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
    cell a = in_5 & in_6;
    cell b = in_1 & in_7;
    cell c = in_1 & in_8;
    cell e = in_0 | in_5;
    cell g = in_1 & in_4;
    cell h = in_6 & in_1;
    cell i = in_7 & in_1;
    cell j = in_8 | in_2;
    cell k = in_2 & in_6;
    cell l = in_7 & in_6;
    cell m = in_1 & in_7;
    cell n = in_4 & in_5;
    cell o = in_7 | in_0;
    cell p = in_6 & in_3;
    cell q = in_6 | in_3;
    cell r = in_2 & in_1;
    cell s = in_6 ^ in_1;
    cell t = in_4 | in_5;
    cell u = m | in_5;
    cell v = in_3 | in_8;
    cell w = in_3 | a;
    cell x = in_2 & p;
    cell y = h | i;
    cell z = in_3 | c;
    cell aa = j | in_2;
    cell ab = in_2 | in_4;
    cell ac = in_3 | in_4;
    cell ae = in_5 | in_0;
    cell ag = in_4 | in_0;
    cell ah = n & in_5;
    cell ai = g | in_5;
    cell aj = in_1 | in_0;
    cell ak = in_5 & in_4;
    cell al = in_7 | in_1;
    cell am = in_2 | in_8;
    cell an = in_3 | in_5;
    cell ao = l | in_8;
    cell ap = ah | b;
    cell aq = r | in_0;
    cell ar = 0;
    cell as = in_4 | in_0;
    cell at = in_4 | ao;
    cell au = u & v;
    cell av = ac | ae;
    cell aw = in_6 & in_2;
    cell ax = aa | in_6;
    cell ay = al | c;
    cell az = in_7 | in_1;
    cell ba = in_0 & in_8;
    cell bb = ae | in_8;
    cell bc = in_1 & in_5;
    cell be = ak & ~in_0;
    cell bg = ai | in_0;
    cell bh = in_5 & in_7;
    cell bi = t & aj;
    cell bj = in_2 & ~in_0;
    cell bk = o & in_1;
    cell bl = az | in_1;
    cell bm = in_2 ^ az;
    cell bn = v & y;
    cell bo = x & q;
    cell bp = in_0 | in_3;
    cell bq = w & as;
    cell br = ax & z;
    cell bs = au | in_0;
    cell bt = an | ba;
    cell bu = in_0 | au;
    cell bv = in_3 & e;
    cell bw = in_4 | aw;
    cell bx = ap | in_2;
    cell by = bp & ay;
    cell bz = bl | in_1;
    cell ca = ab & in_8;
    cell cb = bg & in_4;
    cell cc = bq ^ br;
    cell ce = br & bs;
    cell cg = in_0 | in_5;
    cell ch = in_8 & bt;
    cell ci = in_7 | bv;
    cell cj = bx | in_2;
    cell ck = bw | bu;
    cell cl = bn & y;
    cell cm = in_0 & in_5;
    cell cn = bk & in_1;
    cell co = bz | bo;
    cell cp = bj | in_2;
    cell cq = bi | ca;
    cell cr = bc | ch;
    cell cs = s | bc;
    cell ct = in_2 & bb;
    cell cu = cc | i;
    cell cv = in_1 ^ in_5;
    cell cw = ce | k;
    cell cx = bv | ay;
    cell cy = ch | be;
    cell cz = cl & y;
    cell ea = cp | in_2;
    cell eb = aq | cr;
    cell ec = cv | bs;
    cell ee = in_3 | z;
    cell eg = cj & in_5;
    cell eh = in_1 & bt;
    cell ei = cb | bm;
    cell ej = ct & bb;
    cell ek = ck | ay;
    cell el = bu & in_5;
    cell em = in_2 & ag;
    cell en = eh & in_3;
    cell eo = cs & cw;
    cell ep = cw & q;
    cell eq = bb | by;
    cell er = cm & in_5;
    cell es = ea | in_6;
    cell et = cx | cz;
    cell eu = ei | in_5;
    cell ev = in_6 | ej;
    cell ew = ek | el;
    cell ex = in_0 & am;
    cell ey = cq | em;
    cell ez = cu & in_7;
    cell ga = ep & eq;
    cell gb = eu | ar;
    cell gc = in_3 & ec;
    cell ge = am & et;
    cell gg = cn | ex;
    cell gh = in_8 & es;
    cell gi = cy | cn;
    cell gj = eg | by;
    cell gk = gg & at;
    cell gl = ey | ez;
    cell gm = el | gh;
    cell gn = in_6 & av;
    cell go = gb | in_5;
    cell gp = cz & cg;
    cell gq = gj | gc;
    cell gr = ci | er;
    cell gs = ba & co;
    cell gt = eb & cn;
    cell gu = gq | gk;
    cell gv = gn & go;
    cell gw = ee & in_3;
    cell gx = gu | gv;
    cell gy = eo | ez;
    cell gz = gm & bt;
    cell ha = gi | in_2;
    cell hb = ga & co;
    cell hc = ew | gz;
    cell he = gl | gs;
    cell hg = gc | gy;
    cell hh = gx | ge;
    cell hi = gt | cr;
    cell hj = hh | ge;
    cell hk = hj | gw;
    cell hl = ge | en;
    cell hm = hi | hb;
    cell hn = he | bh;
    cell ho = hl | gy;
    cell hp = hg & ha;
    cell hq = hn | in_6;
    cell hr = hk | gh;
    cell hs = gr | hg;
    cell ht = hm & ev;
    cell hu = gp | ~hr;
    cell hv = hq | in_6;
    cell hw = ~(hb ^ hc);
    cell hx = ht & hs;
    cell hy = hv | hp;
    cell hz = hu | ~hy;
    cell ia = ho | gy;
    cell ib = er & ia;
    cell ic = ~hx;
    cell ie = ic & ~hw;
    cell ig = ie & ~hz;
    cell out = ig & ~ib;
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
        s_out->cells[i] = conway_clayton(in);
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
