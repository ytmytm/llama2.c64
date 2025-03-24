/* Text UI for C64 */

// Maciej 'YTM/Elysium' Witkowiak, 2025

#include <stdio.h>

#include <c64/cia.h>
#include <conio.h>
#include <c64/charwin.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>

//    https://github.com/drmortalwombat/oscar64/blob/main/include/c64/charwin.h

void putbcd(uint8_t bcd) {
    putch((bcd >> 4)   + '0');
    putch((bcd & 0x0f) + '0');
}

void clock_init(void) {
    #define palflag (*((uint8_t *)0x2A6))
    uint8_t tv_mode = palflag << 7;

    // setup clock
    cia2.cra = (cia2.cra & 0x7f ) | tv_mode;
    cia2.todh = 0;
    cia2.todm = 0;
    cia2.tods = 0;
    cia2.todt = 0;
    // force read to start clock
    volatile uint8_t todt = cia2.todt;

}

void clock_display(void) {
    char x = wherex();
    char y = wherey();
    gotoxy(40-8, 0);
    putbcd(cia2.todh); putch(':'); putbcd(cia2.todm); putch(':'); putbcd(cia2.tods);
    // force read to resume register update
    volatile uint8_t todt = cia2.todt;
    gotoxy(x, y);
}

// quasiframes
void ui_quasi_frame(uint8_t top, uint8_t bot, const char* title) {
    char x = wherex();
    char y = wherey();
    textcolor(COLOR_LT_GREY);
    gotoxy(0,top); putch(0xb0);
    for (uint8_t i=1; i<39; i++) putch(0x60);
    putch(0xae);
    gotoxy(2,top);
    textcolor(COLOR_LT_GREEN);
    for (uint8_t i=0; i<strlen(title); i++) putrch(title[i]);
    textcolor(COLOR_LT_GREY);
    gotoxy(0,bot); putch(0xad);
    for (uint8_t i=1; i<39; i++) putch(0x60);
    putch(0xbd);
    gotoxy(x, y);
}

CharWin w_topstatus;
CharWin w_prompt;
CharWin w_output;
CharWin w_bottom;

char *txt_screen = (((char *)0x0400));

#define UI_STATUS_TOP 0
#define UI_PROMPT_TOP 2
#define UI_PROMPT_HEIGHT 5
#define UI_OUTPUT_TOP (UI_PROMPT_TOP+UI_PROMPT_HEIGHT+2)
#define UI_OUTPUT_HEIGHT 12
#define UI_BOTTOM_TOP 23
#define UI_BOTTOM_HEIGHT 1

void ui_init(void) {

}

float temperature = 0.0;    // 0.0 = greedy deterministic. 1.0 = original. don't set higher
float topp = 0.9;           // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
int steps = 60;            // number of steps to run for

void ui_render_temp(void) {

    if (temperature < 0.0) temperature = 0.0;
    if (temperature > 1.0) temperature = 1.0;

    char x = wherex();
    char y = wherey();
    gotoxy(20,17);
    textcolor(COLOR_WHITE);
    printf("%3.1f", temperature);
    gotoxy(x, y);
}

void ui_render_steps(uint16_t maxsteps) {

    if (steps < 10) steps = 10;
    if (steps > maxsteps) steps = maxsteps;

    char x = wherex();
    char y = wherey();
    gotoxy(20,19);
    textcolor(COLOR_WHITE);
    printf("%d   ", steps);
    uint16_t minutes = (uint32_t)steps * 488 / 60;
    uint16_t hours = minutes / 60;
    minutes = minutes % 60;
    gotoxy(20,21);
    printf("%d:%02d   ", hours, minutes);
    gotoxy(x, y);
}

void ui_startup_screen(Config64 *c) {
    clrscr();
    iocharmap(IOCHM_PETSCII_1);
    bgcolor(COLOR_BLACK);
    bordercolor(COLOR_BLACK);
    textcolor(COLOR_LT_GREY);
    ui_quasi_frame(0, 3, "");
    gotoxy(15,1); textcolor(COLOR_CYAN); printf("llama2.c64");
    gotoxy(8,2); textcolor(COLOR_LT_BLUE); printf("c64 port by ytm/elysium");
    textcolor(COLOR_LT_GREY);
    ui_quasi_frame(4,14, "MODEL INFORMATION");
    gotoxy(2,6); textcolor(COLOR_GREEN); printf("dimension:");
    gotoxy(20,6); textcolor(COLOR_YELLOW); printf("%d", c->dim);
    gotoxy(2,7); textcolor(COLOR_GREEN); printf("hidden dimension:");
    gotoxy(20,7); textcolor(COLOR_YELLOW); printf("%d", c->hidden_dim);
    gotoxy(2,8); textcolor(COLOR_GREEN); printf("layers:");
    gotoxy(20,8); textcolor(COLOR_YELLOW); printf("%d", c->n_layers);
    gotoxy(2,9); textcolor(COLOR_GREEN); printf("heads:");
    gotoxy(20,9); textcolor(COLOR_YELLOW); printf("%d", c->n_heads);
    gotoxy(2,10); textcolor(COLOR_GREEN); printf("k/v heads:");
    gotoxy(20,10); textcolor(COLOR_YELLOW); printf("%d", c->n_kv_heads);
    gotoxy(2,11); textcolor(COLOR_GREEN); printf("max sequence len:");
    gotoxy(20,11); textcolor(COLOR_YELLOW); printf("%d", c->seq_len);
    gotoxy(2,12); textcolor(COLOR_GREEN); printf("vocabulary size:");
    gotoxy(20,12); textcolor(COLOR_YELLOW); printf("%d", c->vocab_size);
    textcolor(COLOR_LT_GREY);
    ui_quasi_frame(15,23, "PARAMETERS");
    textcolor(COLOR_GREEN);
    gotoxy(2,17); printf("temperature:");
    gotoxy(2,19); printf("output tokens:");
    gotoxy(2,21); printf("estimated time:");
    textcolor(COLOR_LT_GREY);
    gotoxy(27,17); printf("(+/-)");
    gotoxy(27,19); printf("(,/. or </>)");
    textcolor(COLOR_RED);
    gotoxy(8,24); printf("press <return> to start");

    ui_render_steps(c->seq_len);
    ui_render_temp();
    while (1) {
        char ch = getch();
        if (ch == ',') { steps--; ui_render_steps(c->seq_len); }
        if (ch == '.') { steps++; ui_render_steps(c->seq_len); }
        if (ch == '<') { steps-=10; ui_render_steps(c->seq_len); }
        if (ch == '>') { steps+=10; ui_render_steps(c->seq_len); }
        if (ch == '-') { temperature -= 0.1; ui_render_temp(); }
        if (ch == '+') { temperature += 0.1; ui_render_temp(); }
        if (ch == PETSCII_RETURN || ch == 10 ) { break; }
    }
}

void ui_gotooutput(void) {
    gotoxy(0, UI_OUTPUT_TOP);
    printf("\x05"); // white
}

void ui_inference_screen_init(void) {

    clrscr();
    iocharmap(IOCHM_PETSCII_2);

    // full-length windows
    cwin_init(&w_topstatus, txt_screen, 0, UI_STATUS_TOP, 40, 1);
    cwin_init(&w_prompt,    txt_screen, 0, UI_PROMPT_TOP, 40, UI_PROMPT_HEIGHT);
    cwin_init(&w_output,    txt_screen, 0, UI_OUTPUT_TOP, 40, UI_OUTPUT_HEIGHT);
    cwin_init(&w_bottom,    txt_screen, 0, UI_BOTTOM_TOP, 40, UI_BOTTOM_HEIGHT);

    ui_quasi_frame(UI_PROMPT_TOP-1, UI_PROMPT_TOP+UI_PROMPT_HEIGHT, "prompt");
    ui_quasi_frame(UI_OUTPUT_TOP-1, UI_OUTPUT_TOP+UI_OUTPUT_HEIGHT, "output");

    textcolor(COLOR_PURPLE);
    gotoxy(4, 24);
    printf(p"Llama2.c64 by YTM/Elysium (2025)");
    textcolor(COLOR_LT_GREY);
}

void ui_setcurrenttoken(uint16_t pos, uint16_t steps) {
    char buf[8];
    sprintf(buf, "%03d/%03d", pos, steps);
    char x = wherex();
    char y = wherey();
    gotoxy(40-2-7,UI_OUTPUT_TOP-1);
    puts(buf);
    gotoxy(x, y);
    clock_display();
}

void ui_setnumberoftokens(uint16_t n) {
    char buf[4];
    sprintf(buf, "%03d", n);
    char x = wherex();
    char y = wherey();
    gotoxy(40-2-3,UI_PROMPT_TOP-1);
    textcolor(COLOR_YELLOW);
    puts(buf);
    gotoxy(x, y);
}

void ui_cleartopstatus(void) {
    cwin_clear(&w_topstatus);
    clock_display();
}

void ui_settopstatus(const char *msg) {
    cwin_clear(&w_topstatus);
    cwin_cursor_move(&w_topstatus, 0, 0);
    cwin_put_string(&w_topstatus, msg, 3);
    clock_display();
}

char *ui_get_prompt(char *buffer) {
    char r = 0;

    clock_init();
    cia2.todh = 0; // force clock to stop

    ui_settopstatus("enter prompt, <return>");
    while (1) {
        while (r!=PETSCII_RETURN) {
            cwin_clear(&w_prompt);
            r = cwin_edit(&w_prompt);
        }
        cwin_read_string(&w_prompt, buffer);
        if (strlen(buffer) > 0) break;
        r = 0;
    }

    ui_settopstatus("");

    clock_init(); // restart clock

    // Convert buffer from PETSCII to ASCII
    for (uint16_t i = 0; i < strlen(buffer); i++) {
        if ((buffer[i] >= 0x41 && buffer[i] <= 0x5A) || (buffer[i] >= 0x61 && buffer[i] <= 0x7A)) {
            buffer[i] ^= 0x20; // Convert uppercase PETSCII to ASCII
        }
    }

    return buffer;
}
