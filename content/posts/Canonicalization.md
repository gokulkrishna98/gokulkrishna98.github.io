---
title: "Conanicalization \U0001F4A3"
date: 2024-08-25
draft: false
ShowToc: true
# cover:
#     image: "/images/gokul_image.jpeg"
#     responsiveImages: false
---

This article summarizes my understanding of canonical forms from the perspective of intermediate representation (compilers). We also go through how we can use MLIR pattern rewrite to implement canonicalization of operations.
# What is Canonicalization ?
It is a process which converts data (that can have one more possible representation) into a 'standard', 'normal' or 'canonical form'. We can visualize the data through simple arithmetic expression which can have multiple forms:
```
x + 9
x + 5 + 4
3*3 + x
(1 << 3) + 1 + x
```

Canonicalization is essentially means picking one of these forms to be canonical form, then convert all the other ways to write the expression in the program to this canonical form.

The goal of canonicalization is to make future optimization easier to implement. This helps compiler to not worry about expression with different forms. Imagine, we have a optimization involving the expression `x + 9`, we can just look for the canonical form of this expression after canonicalization, instead of worry about all the equivalent forms.

# How do we Choose Canonical Forms ?
This is primary function of Canonicalization. There are lot of reasons one may prefer one form over another, they can be:
- It results in simple and concise representation (its natural to be in this form). `8` is the canonical form of `5+3`, `1<<3`, `2*2*2` because its natural simple and efficient.
- Some forms are chosen for human aesthetics. I prefer `x + 3` over `3 + x`.
- Sometimes the chosen form is fast or optimal on a target machine (for example the first reason results in faster form).

There are lot of trade-offs between different factors. Like Don Gohman described: What is more useful `2*x` or `x + x` ?

> " ... `x * 2` is equivalent to `x + x`; which of these should be the canonical form? It might seem like we might want to say: pick whatever’s optimal for the target architecture. Addition is generally faster than multiplication, so that would suggest we pick `x + x` as the canonical form. But, `x + x` can actually make things harder for subsequent optimizations, because it means that now `x` has multiple uses. Having multiple uses makes some optimizations more complex – in terms of the dependence graph, this is a DAG rather than a tree, and trees are generally simpler to work with. So maybe `x * 2` is actually a better canonical form, even if it’s a worse optimal form. ..... But ultimately, in its purest form, canonicalization just focuses on removing unnecessary variation so that subsequent optimizations can be simpler."

So, the goal of the canonicalization is not to convert expression into optimal form, but choose a form which makes the work of the back-end of the compiler simpler.  The back-end of the compiler's job would be convert these canonical forms to optimal forms for the target machine.

Don Gohman discusses nuanced topics like : redundancy, inlining, excessive canonicalization and compression (read it [here](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html)).

Now let us see how we can implement canonicalization of operation using MLIR rewrite patterns

# Implementing Canonicalization using MLIR Pattern Rewrite
This is my implementation of pattern rewrite using tablegen declarative way based of toy chapter 3. You could find the implementation details via this commit: [here](https://github.com/gokulkrishna98/GGlow/commit/4890bb79cbda8090a315f21e95aa778da620748a)

Reshape Operation Definition:
```tablegen
def ReshapeOp : GGlowOp <"reshape", [Pure]> {
let summary = "reshape operation";
let description = [{
	Reshape operation is transforming its input tensor into a new tensor
	with the same number of elements but different shapes. For example:

	%0 = gglow.reshape (%arg1 : tensor<10xf64>) -> tensor<5x2xf64>

}];
  
let arguments = (ins F64Tensor:$input);
let results = (outs StaticShapeTensorOf<[F64]>);

let assemblyFormat = [{
	`(` $input `:` type($input) attr-dict `)` `->` type(results)
}];
  
let hasCanonicalizer = 1;
}
```
The `let hasCanonicalizer` provides us hook C++ API to register our Tablegen declared optimization and enable it.

Let us consider the reshape operation. This basically takes in a tensor and reshapes into a specific fixed shape (not given as argument to the operation). So, we have following properties like:
- Reshape(Reshape(x)) = Reshape(x).
- We can remove reshape if input tensor and output tensor reshapes are same.
- Constant Operation -> Reshape can be equivalent to Modified Constant Operation

We can write these patterns via rule based dag approach, defined in `PatternBase.td`.  In this example we use a simplified construct called `Pat` which is defined as:
```tablegen
class Pat<
    dag sourcePattern, dag resultPattern,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)> :
	  Pattern<sourcePattern, [resultPattern], additionalConstraints, 
	  benefitAdded>;
```
Each pattern is specified as a TableGen `dag` object with the syntax of `(operator arg0, arg1, ...)`. The operator can be Operations and directives (like replaceWithValue, Constraints etc )

We provide the following information:
- `sourcePattern` : It finds this dag pattern in IR and tries to replace it with `resultPattern`.
- `resultPattern`: The dag canonical form of the operation.
- `additionalConstraints`: list of dag (formed via constraint definition) that provides additional constraints which it must satisfy to use this pattern transformation.
- `benefits added`: It is kind of priority value. If multiple patterns match for an operation, use the one higher benefit value. Default value is 0.

Let see how we implement these three using Table gen:
1. Reshape(Reshape(x)) = Reshape(x)
```tablegen
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;
```

2. If Reshape(x) = x, then remove Reshape Operation. We first check if input and output shapes are same (by checking type equivalence of tensor dialect). Then add it as constraint to our Pat. We replace the operation with argument value.
```tablegen
def TypesIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : 
	Pat <(ReshapeOp:$res $arg),(replaceWithValue $arg),
		[(TypesIdentical $res, $arg)]>;
```

3. Constant Operation -> Reshape can be equivalent to Modified Constant Operation.
```tablegen
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<mlir::ShapedType>())">;

def FoldConstantReshapePattern : Pat <
	(ReshapeOp:$res (ConstantOp $arg)),
	(ConstantOp (ReshapeConstant $arg, $res))>;
```

After this we have to register these patterns into CanonicalizationPatterns Hook, this can be done via overriding function from Tablegen:
```C++
void mlir::gglow::ReshapeOp::getCanonicalizationPatterns(
	mlir::RewritePatternSet &results, mlir::MLIRContext *context)
{
	    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern, 
				    FoldConstantReshapePattern>(context);
}
```

Running Canonicalization on the Parsed MLIR and getting the result IR.
```C++
auto module = mlir::parseSourceString<mlir::ModuleOp>(ir_content,
													  &context);

if (!module){
	llvm::errs() << "Failed to parse MLIR module\n";
	return;
}

if(enableOpt){
	mlir::PassManager pm(module.get()->getName());
	pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
	if (mlir::failed(pm.run(*module)))
		llvm::errs() << "Failed to canonicalize\n";
}

module->dump();
```

Finally this results in simplification of the following IR:
Source:
```C++
auto reshapeop_string = R"(
	module {
		func.func @reshape_simplify() -> tensor<2x1xf64> {
			%0 = gglow.constant(dense<[1.0, 2.0]> : tensor<2xf64>) -> tensor<2xf64>
			%1 = gglow.reshape (%0: tensor<2xf64>) -> tensor<2x1xf64>
			%2 = gglow.reshape (%1: tensor<2x1xf64>) -> tensor<2x1xf64>
			%3 = gglow.reshape (%2: tensor<2x1xf64>) -> tensor<2x1xf64>
			return %3 : tensor<2x1xf64>
		}
	}
)";

```
Result:
```mlir
module {
  func.func @reshape_simplify() -> tensor<2x1xf64> {
    %0 = gglow.constant(dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>) -> tensor<2x1xf64>
    return %0 : tensor<2x1xf64>
  }
}
```

# References
- Canonicalization Wikipedia : [here](https://en.wikipedia.org/wiki/Canonicalization)
- Dan Gohman article on Canonicalization : [here](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html)
- Toy Chapter 3 MLIR : [here](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)