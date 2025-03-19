
#include <c64/cia.h>
#include <conio.h>
#include <c64/charwin.h>
#include <stdio.h>
#include <stdint.h>

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

int main(void) {

    CharWin top;
    CharWin bot;
    char *screen = (((char *)0x0400));

    clrscr();

    cwin_init(&top, screen, 1, 0, 38, 5);
    cwin_init(&bot, screen, 5, 22, 40-10, 2);

    cwin_clear(&top);

    cwin_fill(&bot, 0x21, 1);

    cwin_cursor_move(&bot, 0,0);
    cwin_put_string(&bot, "Hello, World!aaaaaaaaaaaaaaaeeeeeeeeeeeebbbbbbbbbbbbbbbb", 3); // wyjdzie poza okno na ekran
    
    cwin_edit(&top);

    cwin_cursor_move(&bot, 0,0);
    cwin_put_string(&bot, "PROMPT OK? (Y/N)",3);

    uint8_t x = 0;
    while (1) {
        char c = getch();
        if (c == 'Y' || c == 'y') {
            x=1;
            break;
        }
        if (c == 'N' || c == 'n') {
            break;
        }
    }

    if (x) {
        cwin_cursor_move(&bot, 0,0);
        cwin_put_string(&bot, "PROMPT OK!        ",3);
    } else {
        cwin_cursor_move(&bot, 0,0);
        cwin_put_string(&bot, "PROMPT CANCELLED! ",3);
    }

    clock_init();
    while (1) {
        clock_display();
        if (getchx()) break;
    }
}