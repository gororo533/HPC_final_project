/* This file has been generated with opag 0.6.4, extended for top-k option. */

#ifndef HDR_PARSE
#define HDR_PARSE 1

extern int parse_commandline(int argc, char *argv[]);

/* Top-K extension: scans argv for --topk N or -T N */
void parse_topk_option(int argc, char *argv[]);

#endif
