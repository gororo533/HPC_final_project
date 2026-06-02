# Makefile for ACOTSP (HPC-modified version)
VERSION = 1.03

# ----- Compiler & flags -----
CC       = gcc
# Upgrade from -ansi to C99; keep -Wall, add useful warnings
WARN_FLAGS  = -Wall -Wextra -Wshadow -Wno-unused-parameter
STD_FLAGS   = -std=c99
OPTIM_FLAGS = -O2
CFLAGS      = $(STD_FLAGS) $(WARN_FLAGS) $(OPTIM_FLAGS)
LDLIBS      = -lm

# ----- Timer backend -----
# Use unix_timer on Linux/macOS (measures real + virtual/CPU time separately)
# Fall back to dos_timer on Windows
TIMER = unix
#TIMER = dos

# ----- Sources -----
OBJS = acotsp.o TSP.o utilities.o ants.o InOut.o $(TIMER)_timer.o ls.o parse.o topk.o

# ----- Targets -----
.PHONY: all clean

all: acotsp

acotsp: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

acotsp.o:     acotsp.c     ants.h utilities.h InOut.h TSP.h timer.h ls.h topk.h
TSP.o:        TSP.c        TSP.h InOut.h ants.h ls.h utilities.h
ants.o:       ants.c       ants.h InOut.h TSP.h ls.h utilities.h timer.h
InOut.o:      InOut.c      InOut.h TSP.h timer.h utilities.h ants.h ls.h parse.h topk.h
utilities.o:  utilities.c  utilities.h InOut.h TSP.h ants.h timer.h
ls.o:         ls.c         ls.h ants.h InOut.h TSP.h utilities.h
parse.o:      parse.c      parse.h InOut.h utilities.h ants.h ls.h topk.h
topk.o:       topk.c       topk.h ants.h utilities.h InOut.h TSP.h timer.h
$(TIMER)_timer.o: $(TIMER)_timer.c timer.h

clean:
	@$(RM) *.o acotsp
