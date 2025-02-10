---
title: "RISC-V, a brief learning"
date: 2025-02-05
draft: false
ShowToc: true
---

Reduced Instructor Set Computer version 5 (RISC - V), is an open instruction set architecture. It supports both 32-bit and 64-bit address space. In this article, we will be learning the 32 bit version. We will be learning RV32I, before that let us explore hardware terms, execution environment and general overview.

## Hardware Terminology
- **Core**: A hardware component is called core if it contains an independent instruction fetch unit.
- **Hart**: Each instruction fetch is performed by a hardware thread on cores. Each core can execute multiple hardware threads (using hyper-threading, multi-threading etc.)
- **Coprocessor:** It is a hardware unit attached to core which has additional architectural state and implements the instruction extensions.
- **Accelerator:** It is a core that can operate autonomously but is specialized for certain tasks. Example: I/O processor which can offload IO tasks from core to these units.
## Execution Environment 

It defines how software interacts with the hardware and system software, ensuring compatibility across different RISC-V implementations. This is done to standardize execution models, system calls, and memory models to enable smooth operation of applications, operating systems, and hypervisors. The RISC-V supports different execution environments based on needs:
- **Bare metal execution:** The program runs directly on hardware and application can directly access hardware. The harts are implemented by physical processor threads and instructions have full access to address space.
- **Operating System:** The RISC-V OS executes multiple user-level execution environments by multiplexing user-level harts onto available physical processor threads and by controlling access to memory via virtual memory.
- The RISC-V also has support for hypervisors and emulators (like Qemu, spike etc.)

![](/images/6b94cc06b31dcf3151ada8adf5512058.png)

## General Overview
A RISC-V ISA is defined as a base integer ISA, which must be present in any implementation, plus optional extensions to the base ISA. It is similar to RISC instruction set, but it supports variable length instruction encoding.

RISC-V currently has 4 base ISAs, where each ISA is described by the size of address space and number of integer registers : RV32I, RV64I, RV32E and RV64E. `E` has half the number of integer registers as `I` variant. We use the term `XLEN` to define the width of the registers.

Note: RV32I is not a subset RV64I, this is because the 64 bit version does need to encode instructions specialized for 32 bit version. But this complicates the implementation of 32bit version on 64bit. 

The base integer ISA is named "I" (prefixed by RV32 or RV64 depending on integer register width), and contains integer computational instructions, integer loads, integer stores, and control-flow instructions. There are a lot of extensions like -
- M: multiply and divide.
- A: for atomic operations.
- F: floating point operations,
- C: compressed instructions.

In general word = 32bit and memory address is circular, that is after $2^{XLEN-1}$, we go to 0 address (so we can use modulo to do some complex stuff). 

The execution environment determines the mapping of hardware resources into a hart’s address space. Different address ranges of a hart’s address space may (1) be vacant, or (2) contain main memory, or (3) contain one or more I/O devices. Reads and writes of I/O devices may have visible side effects, but accesses to main memory cannot. 

Note: Ordinarily, if an instruction attempts to access memory at an inaccessible address, an exception is raised for the instruction. Vacant locations in the address
space are never accessible.

The compressed or expanded instructions are parceled as 16-bit data (alignment considered) packets. This alignment is defined using $IALIGN$ (for base = 32, for compressed = 16). The RISC-V encoding supports 32 bit, 48 bit, 64 bit, 80-192 bit and as of now >=192 bit is reserved. The following table defines how these are indicated:

![](/images/009d9f82815206e969915bab6d8261cb.png)
There are few terms, which is useful for OS and CA which are used a lot:
- **Exceptions**: Unusual condition occurring at run time associated with an instruction in the current RISC-V hart.
- **Interrupt**: an external asynchronous event that may cause a RISC-V hart to experience an unexpected transfer of control.
- **Traps**: It refers to transfer of control to a trap handler caused by either an exception or an interrupt.

## RV32 - Base Integer ISA

In this ISA, there are 32 registers from x0 to x32. The x0 is hardwired to be all zero.
There is no dedicated stack pointer or subroutine return address link register in the Base Integer ISA; the instruction encoding allows any x register to be used for these purposes. However, the standard software calling convention uses register x1 to hold the return address for a call, with register x5 available as an alternate link register. The standard calling convention uses register x2 as the stack pointer.

### Instruction formats
There are four core instruction format namely : R, S, U and I. The RISC-V ISA keeps the source (rs1 and rs2) and destination (rd) registers at the same position in all formats to simplify decoding.

![](/images/7b1de3a8b622f93f93095d1e919f32c8.png)

There are 3 main categories of instructions:
1. Integer computation instruction
	- Immediate - Register instr.
	- Register - Register instr.
	- NOP instr.
2. Control Transfer instruction
	- Unconditional
	- Conditional
3. Load and Store instruction.

### Integer computation instruction

Immediate is a value that can be directly given to instruction, it is not stored in any register. It is encoded in the instruction word. Sign-extended immediate is when we have negative number written in 8 bit format, when operating in 32 bit, the sign bit is moved to preserve the equivalent value. Direct copy will result in value change in 2's complement form.

Note: The computation can exceed the register size i.e. they overflow. The overflow is ignored, but we can check if it is overflow by using comparison instruction on result and the source values.
#### 1. Immediate - Register instruction.
These are the class of instructions in which one of the argument is immediate and other is a register. I majorly categorize them as addition, comparison, bitwise logical, shift and load operation.
- **Addition (addi)**: It adds sign-extended immediate value and a value in register and stores it in another register. The value can be only 12 bit value. Note: The `mov` instruction is basically add instruction with 0 as immediate 
```
addi rd, rs, 0x12
mov rd, rs <=> addi rd, rs, 0
```
- **Comparison (slti or sltiu)**: it compares a value in register and sign-extended immediate value. The u variant treats the immediate value as unsigned number. Note: `seqz` (comparing a register to 0 zero) is implemented using sltiu with 1 as immediate value.
```
slti rd, rs, -12
sltiu rd, rs, 123
seqz rd, rs <=> sltiu rd, rs, 1
```

- **Logical (andi, ori and xori)**: it performs bitwise add between a register and immediate value and stores it in destination register. Note: the logical inversion of bits (bitwise `not`) is implemented using `xori` by keeping immediate value as negative one.
```
andi rd, rs, 0x123
ori rd, rs, 0x123
xori rd, rs, 0x123
not rd, rs <=> xori rd, rs, -1
```

![](/images/eccfcbfd8fe6ab7ff9a6c59d8c34506b.png)

- **Shift (slli, srli and srai)**: it does the bitwise shift (<< or >> in c) to the values in register by a 5 bit immediate value, which stored in lower immediate (20:24). There are generally two shift operation:
	- left shift - slli (shift left logical immediate)
	- right shift - Here the right shift influences the signed bit value, so we have two types of right shift: one logical and one arithmetic. These are differentiated using bit 30.
		- srli - it moves the bits to the right but fills the new bits on the higher bit side with 0s.
		- srai - it moves the bits to the right, but fills leftmost bits with the sign bit (MSB).
![](/images/5286b5d3b9a5cf79a3b16a3a4dee474b.png)
Examples - 
slli:
```
li x5, 0b00000000000000000000000000001101 
slli x6, x5, 2

before:
00000000000000000000000000001101 (13 in decimal)
after:
00000000000000000000000000110100 (52 in decimal, 0x34)
```
srli:
```
li x5, 0b11110000000000000000000000000000
srli x6, x5, 4                    

before:
11110000000000000000000000000000
after:
00001111000000000000000000000000
```
srai
```
li x5, -16        # 0xFFFFFFF0 (2's complement form)
srai x6, x5, 2

before:
11111111111111111111111111110000
after:
11111111111111111111111111111100
```
- **load (lui and auipc)**: These are not the typical load and store instruction related to memory, but it is loading immediate value to a register directly. It can load up to 20 bit immediate values. All the previous integer (register - immediate) instructions uses I type encoding but here we use U type encoding as we need just one destination register and immediate value.
	- lui - it places the 32-bit U-immediate value into the destination register rd, filling in the lowest 12 bits with zeros.
	- auipc - it forms a 32-bit offset from the U-immediate, filling in the lowest 12 bits with zeroes, adds this offset to the address of the AUIPC instruction, then places the result in register rd.
![](/images/efebf948c3b4df10905e07fed6348da8.png)
#### 2. Register - Register Instructions.
These instructions use R-type encoding format. All operations read the `rs1` and `rs2` registers as source operands and write the result into register `rd`. The `funct7` and `funct3` fields select the type of operation.

![](/images/fd8fdc654caf8ef0ee06017e03ab94c2.png)
The only extra instruction is SUB, which subtracts `rs2` from `rs1` and stores in `rd`. This instruction does not exists in immediate version as we have signed immediate notation.

#### 3. NOP Instruction.

The NOP instruction does not change any architecturally visible state, except for advancing the pc and incrementing any applicable performance counters. NOP is encoded as ADDI x0, x0, 0.
![](/images/fb3b4ebb5dc1e4352010a69ec15a1938.png)
### Control Transfer Instruction.
There are two types control transfer instruction: unconditional jump and conditional jump.
#### 1. Unconditional Jump: 
There are two types of unconditional jump: `jal` and `jalr`. The jump and link (jal) instruction uses the J-type format (a extension of U format), where the J-immediate encodes a signed offset in multiples of 2 bytes. The offset is sign-extended and added to the address of the jump instruction to form the jump target address. Jumps can therefore target a ±1 MiB range. JAL stores the address of the instruction following the jump ('pc'+4) into register rd.
![](/images/9fc2ca33c29a828e053316d00a5a401f.png)
Note: Do not know why the immediate values are not in order similar to U instruction type.

The other instruction jalr (jump and link register) uses I instruction encoding format. This instruction calculates the target address is obtained by adding the sign-extended 12-bit I-immediate to the register `rs1`, then setting the least-significant bit of the result to zero.

![](/images/4526856f212e047374711e93f401a7fa.png)
#### 2. Conditional Jump
This instruction uses B type encoding format, which is a slight variation to S type encoding format. In this instruction it compares two registers and based on the result jumps to a particular address given by offset (by immediate value). The 12-bit B-immediate encodes signed offsets in multiples of 2 bytes. The offset is sign-extended and added to the address of the branch instruction to give the target address. The conditional branch range is ±4 KiB.

![](/images/70808101e43e50fcf500a9c9ff84b442.png)

###  Load and Store Instruction.
Load and store instructions transfer a value between the registers and memory. Loads are encoded in the I-type format and stores are S-type. In both the instructions, we get the memory address by adding immediate with `rs1`. 
Example:
```
lw rd, offset(rs1)

Note:
- rd → Destination register where the loaded word will be stored.
- offset → A 12-bit signed immediate value, representing the memory offset in bytes.
- rs1 → Base register containing the memory address to which the offset is added.
```

![](/images/5785e06f6f64eae07c5fa781de86e414.png)

This covers the base ISA, the extension are pretty easy to understand and can be read separately.

## References
- RISC-V Reference Manual : https://github.com/riscv/riscv-isa-manual/releases

