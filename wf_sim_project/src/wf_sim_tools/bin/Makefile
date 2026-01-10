# Compiler
CC = gcc

# O2 Cluster Specific Paths
# We define the base path to make it easier to use in multiple places
GSL_PATH = /n/app/gsl/2.8-gcc-14.2.0
GSL_INC = -I$(GSL_PATH)/include
GSL_LIB = -L$(GSL_PATH)/lib

# Linker Flags
# -Wl,-rpath tells the loader to look in this specific directory at runtime
LDFLAGS = -Wl,-rpath,$(GSL_PATH)/lib

# Flags
CFLAGS = -std=c99 -O3 -Wall -Wextra $(GSL_INC)
LIBS = $(GSL_LIB) -lgsl -lgslcblas -lm

# Target name
TARGET = wf_sim
SRC = sim.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(TARGET)