# Compiler
COMPILER ?= g++
NVCC ?= nvcc


# Base flags
CXXFLAGS := -w -march=native -Ofast -fno-finite-math-only -std=c++20 -pthread -I SimpleNN
NVCCFLAGS := -Xptxas -O3

# Additional flags for overwriting macros
MACRO_FLAGS :=

# Precompiled header and config file
PCH := include/pch.h
PCH_OBJ := $(PCH:.h=.gch)
CONFIG := config.h

# Check if USE_CUDA_GEMM is defined, otherwise take from config.h
USE_CUDA_GEMM := $(shell grep -oP '(?<=define USE_CUDA_GEMM )\d+' $(CONFIG))
PROTOCOL := $(shell grep -oP '(?<=define PROTOCOL )\d+' $(CONFIG))
PARTY := $(shell grep -oP '(?<=define PARTY )\d+' $(CONFIG))
SPLITROLES ?= 0


# List of all possible configuration options
CONFIG_OPTIONS := BASE_PORT BITLENGTH NUM_INPUTS DATTYPE FUNCTION_IDENTIFIER COMPRESS \
                  LIVE INIT OPT_SHARE PROTOCOL PREPROCESSING USE_SSL PROCESS_NUM \
                  RANDOM_ALGORITHM CONNECTION_TIMEOUT CONNECTION_RETRY SEND_BUFFER \
                  RECV_BUFFER VERIFY_BUFFER USE_CUDA_GEMM

# Targets
.PHONY: all clean compile_pch compile_parties link_objects

all: compile_pch compile_executables link_objects

compile_pch:
	@if [ ! -f $(PCH_OBJ) ] || [ $(PCH) -nt $(PCH_OBJ) ]; then \
		echo "Compiling precompiled header..."; \
		$(COMPILER) $(CXXFLAGS) -x c++-header $(PCH) -o $(PCH_OBJ); \
	else \
		echo "Precompiled header is up to date."; \
	fi

define update_config
	$(eval MACRO_FLAGS := $(shell for option in $(CONFIG_OPTIONS); do value=$${!option}; if [ ! -z "$$value" ]; then echo -D$$option=$$value; fi; done | tr '\n' ' '))
endef


compile_executables:
	@if [ "$(SPLITROLES)" -eq 3 ]; then \
		$(MAKE) compile_splitroles_3; \
	elif [ "$(SPLITROLES)" -eq 34 ]; then \
		$(MAKE) compile_splitroles_3to4; \
	elif [ "$(SPLITROLES)" -eq 4 ]; then \
		$(MAKE) compile_splitroles_4; \
	else \
		$(MAKE) compile_parties; \
	fi

compile_parties:
	@if [ "$(PARTY)" = "all" ]; then \
		if [ $(PROTOCOL) -gt 6 ]; then \
			$(MAKE) compile_player_0,1,run-P0 compile_player_1,1,run-P1 compile_player_2,1,run-P2 compile_player_3,1,run-P3; \
		else \
			$(MAKE) compile_player_0,1,run-P0 compile_player_1,1,run-P1 compile_player_2,1,run-P2; \
		fi; \
	else \
		$(MAKE) compile_player_$(PARTY),1,run-P$(PARTY); \
	fi



compile_player_%:
	$(MAKE) do_compile_player_$* PARTY_ARGS=$*

do_compile_player_%:
	$(eval LPARTY := $(shell echo '$(PARTY_ARGS)' | cut -d',' -f1))
	$(eval SPLIT_ROLES_OFFSET := $(shell echo '$(PARTY_ARGS)' | cut -d',' -f2))
	$(eval EXEC_NAME := $(shell echo '$(PARTY_ARGS)' | cut -d',' -f3-))
	$(update_config)
	@PREV_MACRO_FLAGS_FILE=./flags/$(EXEC_NAME).macro_flags; \
	CURRENT_FLAGS="$(MACRO_FLAGS) -DPARTY=$(LPARTY) -DSPLIT_ROLES_OFFSET=$(SPLIT_ROLES_OFFSET)"; \
	if [ -f ./$(EXEC_NAME).o ] && [ -f $$PREV_MACRO_FLAGS_FILE ] && cmp -s <(echo "$$CURRENT_FLAGS") $$PREV_MACRO_FLAGS_FILE && [ ./$(EXEC_NAME).o -nt main.cpp ] && [ ./$(EXEC_NAME).o -nt $(PCH) ] && [ ./$(EXEC_NAME).o -nt $(CONFIG) ] && [ ./$(EXEC_NAME).o -nt $(PCH_OBJ) ]; then \
		echo "Nothing to do for $(EXEC_NAME)"; \
	else \
		echo "Compiling executable $(EXEC_NAME)" ; \
		if [ $(USE_CUDA_GEMM) -gt 0 ]; then \
			$(COMPILER) main.cpp -include $(PCH) $(CXXFLAGS) $(MACRO_FLAGS) -DPARTY=$(LPARTY) -DSPLIT_ROLES_OFFSET=$(SPLIT_ROLES_OFFSET) -c -o ./$(EXEC_NAME)-cuda.o; \
			echo "Linking CUDA executable $(EXEC_NAME)" ; \
			case "$(USE_CUDA_GEMM)" in \
			1|3) \
				$(NVCC) ./$(EXEC_NAME)-cuda.o $(NVCCFLAGS) ./cuda/gemm_cutlass_int.o -o ./$(EXEC_NAME).o;; \
			2) \
				$(NVCC) ./$(EXEC_NAME)-cuda.o $(NVCCFLAGS) ./cuda/conv_cutlass_int_NCHW.o -o ./$(EXEC_NAME).o;; \
			4) \
				$(NVCC) ./$(EXEC_NAME)-cuda.o $(NVCCFLAGS) ./cuda/conv_cutlass_int_CHWN.o -o ./$(EXEC_NAME).o;; \
			esac; \
			rm -f ./$(EXEC_NAME)-cuda.o; \
		else \
			$(COMPILER) main.cpp -include $(PCH) $(CXXFLAGS) $(MACRO_FLAGS) -DPARTY=$(LPARTY) -DSPLIT_ROLES_OFFSET=$(SPLIT_ROLES_OFFSET) -o ./$(EXEC_NAME).o; \
		fi; \
		echo "$$CURRENT_FLAGS" > $$PREV_MACRO_FLAGS_FILE; \
		echo "Compilation for xecutable $(EXEC_NAME) completed."; \
	fi



compile_splitroles_3:
	@if [ "$(PARTY)" = "0" ]; then \
		$(MAKE) compile_player_0,0,run-P0--0-1-2 compile_player_0,1,run-P0--0-2-1 compile_player_1,2,run-P0--1-0-2 compile_player_2,3,run-P0--1-2-0 compile_player_1,4,run-P0--2-0-1 compile_player_2,5,run-P0--2-1-0; \
	elif [ "$(PARTY)" = "1" ]; then \
		$(MAKE) compile_player_1,0,run-P1--0-1-2 compile_player_2,1,run-P1--0-2-1 compile_player_0,2,run-P1--1-0-2 compile_player_0,3,run-P1--1-2-0 compile_player_2,4,run-P1--2-0-1 compile_player_1,5,run-P1--2-1-0; \
	elif [ "$(PARTY)" = "2" ]; then \
		$(MAKE) compile_player_2,0,run-P2--0-1-2 compile_player_1,1,run-P2--0-2-1 compile_player_2,2,run-P2--1-0-2 compile_player_1,3,run-P2--1-2-0 compile_player_0,4,run-P2--2-0-1 compile_player_0,5,run-P2--2-1-0; \
	elif [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_0,0,run-P0--0-1-2 compile_player_0,1,run-P0--0-2-1 compile_player_1,2,run-P0--1-0-2 compile_player_2,3,run-P0--1-2-0 compile_player_1,4,run-P0--2-0-1 compile_player_2,5,run-P0--2-1-0 compile_player_1,0,run-P1--0-1-2 compile_player_2,1,run-P1--0-2-1 compile_player_0,2,run-P1--1-0-2 compile_player_0,3,run-P1--1-2-0 compile_player_2,4,run-P1--2-0-1 compile_player_1,5,run-P1--2-1-0 compile_player_2,0,run-P2--0-1-2 compile_player_1,1,run-P2--0-2-1 compile_player_2,2,run-P2--1-0-2 compile_player_1,3,run-P2--1-2-0 compile_player_0,4,run-P2--2-0-1 compile_player_0,5,run-P2--2-1-0; \
	fi
		


compile_splitroles_3to4:
	@if [ "$(PARTY)" = "0" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_0,0,run-P1--1-2-3 compile_player_0,1,run-P1--1-3-2 compile_player_1,2,run-P1--2-1-3 compile_player_2,3,run-P1--2-3-1 compile_player_1,4,run-P1--3-1-2 compile_player_2,5,run-P1--3-2-1 compile_player_0,6,run-P1--1-2-4 compile_player_0,7,run-P1--1-4-2 compile_player_1,8,run-P1--2-1-4 compile_player_2,9,run-P1--2-4-1 compile_player_1,10,run-P1--4-1-2 compile_player_2,11,run-P1--4-2-1 compile_player_0,12,run-P1--1-3-4 compile_player_0,13,run-P1--1-4-3 compile_player_1,14,run-P1--3-1-4 compile_player_2,15,run-P1--3-4-1 compile_player_1,16,run-P1--4-1-3 compile_player_2,17,run-P1--4-3-1; \
	fi
	@if [ "$(PARTY)" = "1" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_1,0,run-P2--1-2-3 compile_player_2,1,run-P2--1-3-2 compile_player_0,2,run-P2--2-1-3 compile_player_0,3,run-P2--2-3-1 compile_player_2,4,run-P2--3-1-2 compile_player_1,5,run-P2--3-2-1 compile_player_1,6,run-P2--1-2-4 compile_player_2,7,run-P2--1-4-2 compile_player_0,8,run-P2--2-1-4 compile_player_0,9,run-P2--2-4-1 compile_player_2,10,run-P2--4-1-2 compile_player_1,11,run-P2--4-2-1 compile_player_0,18,run-P2--2-3-4 compile_player_0,19,run-P2--2-4-3 compile_player_1,20,run-P2--3-2-4 compile_player_2,21,run-P2--3-4-2 compile_player_1,22,run-P2--4-2-3 compile_player_2,23,run-P2--4-3-2; \
	fi
	@if [ "$(PARTY)" = "2" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_2,0,run-P3--1-2-3 compile_player_1,1,run-P3--1-3-2 compile_player_2,2,run-P3--2-1-3 compile_player_1,3,run-P3--2-3-1 compile_player_0,4,run-P3--3-1-2 compile_player_0,5,run-P3--3-2-1 compile_player_1,12,run-P3--1-3-4 compile_player_2,13,run-P3--1-4-3 compile_player_0,14,run-P3--3-1-4 compile_player_0,15,run-P3--3-4-1 compile_player_2,16,run-P3--4-1-3 compile_player_1,17,run-P3--4-3-1 compile_player_1,18,run-P3--2-3-4 compile_player_2,19,run-P3--2-4-3 compile_player_0,20,run-P3--3-2-4 compile_player_0,21,run-P3--3-4-2 compile_player_2,22,run-P3--4-2-3 compile_player_1,23,run-P3--4-3-2; \
	fi
	@if [ "$(PARTY)" = "3" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_2,6,run-P4--1-2-4 compile_player_1,7,run-P4--1-4-2 compile_player_2,8,run-P4--2-1-4 compile_player_1,9,run-P4--2-4-1 compile_player_0,10,run-P4--4-1-2 compile_player_0,11,run-P4--4-2-1 compile_player_2,12,run-P4--1-3-4 compile_player_1,13,run-P4--1-4-3 compile_player_2,14,run-P4--3-1-4 compile_player_1,15,run-P4--3-4-1 compile_player_0,16,run-P4--4-1-3 compile_player_0,17,run-P4--4-3-1 compile_player_2,18,run-P4--2-3-4 compile_player_1,19,run-P4--2-4-3 compile_player_2,20,run-P4--3-2-4 compile_player_1,21,run-P4--3-4-2 compile_player_0,22,run-P4--4-2-3 compile_player_0,23,run-P4--4-3-2; \
	fi



compile_splitroles_4:
	@if [ "$(PARTY)" = "0" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_0,0,run-P1--1-2-3-4 compile_player_0,1,run-P1--1-3-2-4 compile_player_1,2,run-P1--2-1-3-4 compile_player_2,3,run-P1--2-3-1-4 compile_player_1,4,run-P1--3-1-2-4 compile_player_2,5,run-P1--3-2-1-4 compile_player_0,6,run-P1--1-2-4-3 compile_player_0,7,run-P1--1-4-2-3 compile_player_1,8,run-P1--2-1-4-3 compile_player_2,9,run-P1--2-4-1-3 compile_player_1,10,run-P1--4-1-2-3 compile_player_2,11,run-P1--4-2-1-3 compile_player_0,12,run-P1--1-3-4-2 compile_player_0,13,run-P1--1-4-3-2 compile_player_1,14,run-P1--3-1-4-2 compile_player_2,15,run-P1--3-4-1-2 compile_player_1,16,run-P1--4-1-3-2 compile_player_2,17,run-P1--4-3-1-2 compile_player_3,18,run-P1--2-3-4-1 compile_player_3,19,run-P1--2-4-3-1 compile_player_3,20,run-P1--3-2-4-1 compile_player_3,21,run-P1--3-4-2-1 compile_player_3,22,run-P1--4-2-3-1 compile_player_3,23,run-P1--4-3-2-1; \
	fi
	@if [ "$(PARTY)" = "1" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_1,0,run-P2--1-2-3-4 compile_player_2,1,run-P2--1-3-2-4 compile_player_0,2,run-P2--2-1-3-4 compile_player_0,3,run-P2--2-3-1-4 compile_player_2,4,run-P2--3-1-2-4 compile_player_1,5,run-P2--3-2-1-4 compile_player_1,6,run-P2--1-2-4-3 compile_player_2,7,run-P2--1-4-2-3 compile_player_0,8,run-P2--2-1-4-3 compile_player_0,9,run-P2--2-4-1-3 compile_player_2,10,run-P2--4-1-2-3 compile_player_1,11,run-P2--4-2-1-3 compile_player_0,18,run-P2--2-3-4-1 compile_player_0,19,run-P2--2-4-3-1 compile_player_1,20,run-P2--3-2-4-1 compile_player_2,21,run-P2--3-4-2-1 compile_player_1,22,run-P2--4-2-3-1 compile_player_2,23,run-P2--4-3-2-1 compile_player_3,12,run-P2--1-3-4-2 compile_player_3,13,run-P2--1-4-3-2 compile_player_3,14,run-P2--3-1-4-2 compile_player_3,15,run-P2--3-4-1-2 compile_player_3,16,run-P2--4-1-3-2 compile_player_3,17,run-P2--4-3-1-2; \
	fi
	@if [ "$(PARTY)" = "2" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_2,0,run-P3--1-2-3-4 compile_player_1,1,run-P3--1-3-2-4 compile_player_2,2,run-P3--2-1-3-4 compile_player_1,3,run-P3--2-3-1-4 compile_player_0,4,run-P3--3-1-2-4 compile_player_0,5,run-P3--3-2-1-4 compile_player_1,12,run-P3--1-3-4-2 compile_player_2,13,run-P3--1-4-3-2 compile_player_0,14,run-P3--3-1-4-2 compile_player_0,15,run-P3--3-4-1-2 compile_player_2,16,run-P3--4-1-3-2 compile_player_1,17,run-P3--4-3-1-2 compile_player_1,18,run-P3--2-3-4-1 compile_player_2,19,run-P3--2-4-3-1 compile_player_0,20,run-P3--3-2-4-1 compile_player_0,21,run-P3--3-4-2-1 compile_player_2,22,run-P3--4-2-3-1 compile_player_1,23,run-P3--4-3-2-1 compile_player_3,6,run-P3--1-2-4-3 compile_player_3,7,run-P3--1-4-2-3 compile_player_3,8,run-P3--2-1-4-3 compile_player_3,9,run-P3--2-4-1-3 compile_player_3,10,run-P3--4-1-2-3 compile_player_3,11,run-P3--4-2-1-3; \
	fi
	@if [ "$(PARTY)" = "3" ] || [ "$(PARTY)" = "all" ]; then \
		$(MAKE) compile_player_2,6,run-P4--1-2-4-3 compile_player_1,7,run-P4--1-4-2-3 compile_player_2,8,run-P4--2-1-4-3 compile_player_1,9,run-P4--2-4-1-3 compile_player_0,10,run-P4--4-1-2-3 compile_player_0,11,run-P4--4-2-1-3 compile_player_2,12,run-P4--1-3-4-2 compile_player_1,13,run-P4--1-4-3-2 compile_player_2,14,run-P4--3-1-4-2 compile_player_1,15,run-P4--3-4-1-2 compile_player_0,16,run-P4--4-1-3-2 compile_player_0,17,run-P4--4-3-1-2 compile_player_2,18,run-P4--2-3-4-1 compile_player_1,19,run-P4--2-4-3-1 compile_player_2,20,run-P4--3-2-4-1 compile_player_1,21,run-P4--3-4-2-1 compile_player_0,22,run-P4--4-2-3-1 compile_player_0,23,run-P4--4-3-2-1 compile_player_3,0,run-P4--1-2-3-4 compile_player_3,1,run-P4--1-3-2-4 compile_player_3,2,run-P4--2-1-3-4 compile_player_3,3,run-P4--2-3-1-4 compile_player_3,4,run-P4--3-1-2-4 compile_player_3,5,run-P4--3-2-1-4; \
	fi


clean:
	@echo "Cleaning up compiled files..."
	rm -f *.o *.gch run-P*.o run-P*.macro_flags include/*.gch
	@echo "Cleanup completed."

