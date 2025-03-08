dd if=stories260K.bin of=weights.bin bs=1 skip=28
dd if=/dev/zero bs=1040640 count=1 | cat >> weights.bin
