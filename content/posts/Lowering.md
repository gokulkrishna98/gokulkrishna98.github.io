---
title: "Lowering in MLIR"
date: 2024-12-18
draft: false
ShowToc: true
params:
  math: true
# cover:
#     image: "/images/gokul_image.jpeg"
#     responsiveImages: false
---

# Introduction
First let us understand the definition of lowering:

> The process of transforming a higher-level representation of an operation into a lower-level, but 
> semantically equivalent, representation

It means we have a representation representing high level abstraction and we convert it into 
low level abstraction but it should be computationally equivalent (i.e. same meaning or it cannot 
result in different result). Why are we doing this ? Because entire computer ecosystem is built 
this way, we write code in programming languages that gets converted to assembly or byte-code 
which eventually gets converted to 1s and 0s. Generally, in most cases LLVM IR is considered 
lowest level as we have standard compilers to do further lowering. 

In MLIR, there is a concept of `transitive lowering`, it is an A->B->C lowering, in which multiple 
patterns may be applied in order to fully transform an illegal operation into a set of legal ones.

![lowering](/images/abc_lowering.png)

In our case, we treat high level operations like multiplication, convolution as high level operations
 and dialect which defines these are high level dialect (the A dialect in diagram). 
 The we convert these high level operations to be converted to a set of  standard 
 dialects in mlir (like scf, affine, memref). These are intermediate dialects, 
 the B dialect in diagram, the reason we are doing this because we have predefined pattern-rewrites
  for converting these dialects to llvm dialect. The llvm-dialect is our low-level dialect 
  (C in the diagram). We can use the JIT (defined by MLIR) to execute llvm-dialect in the module. 
  This article will discuss three main parts:
- Converting high level dialect to standard (intermediate) dialect.
- Convert standard dialect to llvm-dialect.
- Set up JIT to execute the llvm-dialect.

Note: using the term standard dialect, instead of intermediate as it makes more sense.
Note: MLIR gives option to make specific operation to be legal or illegal. So, when i mention dialect (it includes dialects 😀)
# How to lower a Dialect?
This is a quick overview of how to convert a high level dialect to standard one. The general steps remain same except the creating new operations, which depends on the standard dialect.

MLIR provides us with [dialect conversion](https://mlir.llvm.org/docs/DialectConversion/) framework, which converts illegal dialects to legal ones. There are three things required by:
1. Conversion Target : formal specification of which dialects are legal.
2. A set of rewrite patterns.
3. (Optional) Type converter.

We will be using `PassWrapper` class to implement this lowering.