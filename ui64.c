/* Text UI for C64 */

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
    gotoxy(0,top); putch(0xb0);
    for (uint8_t i=1; i<39; i++) putch(0x60);
    putch(0xae);
    gotoxy(2,top);
    for (uint8_t i=0; i<strlen(title); i++) putrch(title[i]);
    gotoxy(0,bot); putch(0xad);
    for (uint8_t i=1; i<39; i++) putch(0x60);
    putch(0xbd);
    gotoxy(x, y);
}

CharWin w_topstatus;
CharWin w_prompt;
CharWin w_output;

char *txt_screen = (((char *)0x0400));

#define UI_STATUS_TOP 0
#define UI_PROMPT_TOP 2
#define UI_PROMPT_HEIGHT 5
#define UI_OUTPUT_TOP (UI_PROMPT_TOP+UI_PROMPT_HEIGHT+2)
#define UI_OUTPUT_HEIGHT 12

void ui_init(void) {

    iocharmap(IOCHM_PETSCII_2);

    clock_init();

    clrscr();

    // full-length windows
    cwin_init(&w_topstatus, txt_screen, 0, UI_STATUS_TOP, 40, 1);
    cwin_init(&w_prompt,    txt_screen, 0, UI_PROMPT_TOP, 40, UI_PROMPT_HEIGHT);
    cwin_init(&w_output,    txt_screen, 0, UI_OUTPUT_TOP, 40, UI_OUTPUT_HEIGHT);

    ui_quasi_frame(UI_PROMPT_TOP-1, UI_PROMPT_TOP+UI_PROMPT_HEIGHT, "prompt");
    ui_quasi_frame(UI_OUTPUT_TOP-1, UI_OUTPUT_TOP+UI_OUTPUT_HEIGHT, "output");
    clock_display();

}

void ui_gotooutput(void) {
    gotoxy(0, UI_OUTPUT_TOP);
    printf("\x05"); // white
}

void ui_inference_screen_init(void) {

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
    ui_settopstatus("enter prompt, <return>");

    while (r!=13) {
        cwin_clear(&w_prompt);
        r = cwin_edit(&w_prompt);
    }
    ui_settopstatus("");

    // XXX convert from PETSCII to ASCII!
    return cwin_read_string(&w_prompt, buffer);
}

