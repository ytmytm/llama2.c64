
#include <conio.h>
#include <c64/charwin.h>
#include <stdint.h>

//    https://github.com/drmortalwombat/oscar64/blob/main/include/c64/charwin.h

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

}