# Compiling Cython code

Part of the code is written in Cython. To compile it,

```
cython remove_labels.pyx
gcc -O2 -Wall `python3.11-config --include` -o remove_labels.so remove_labels.c
```