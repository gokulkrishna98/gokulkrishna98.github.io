---
title: "In-lining in MLIR"
date: 2024-08-27
draft: false
ShowToc: true
# cover:
#     image: "/images/gokul_image.jpeg"
#     responsiveImages: false
---

In the context of compilers, `inlining or inline expansion` is a process (or optimization depending on the use case) that replaces function call with the body of the function. Now let us see how we can inline a function defined in the IR.

# Prerequisites
Before proceeding, I am implementing these features in my dialect Glow, please find more information [here](https://github.com/gokulkrishna98/GGlow/tree/main/lib/Dialect/GGlow) .

One of the main requirements is defining the function feature in the dialect, we will be utilizing traits and tablegen to implement these.

## 1. Function Operation
This will represent a operation that will return a value. Since, MLIR provides nested structure of Region -> Block -> Operation -> Region ... , we can have a isolated set of computation represented by other operations and returns a value.

```tablegen
def FuncOp : GGlowOp<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "user defined function operation";
    let description = [{
        The "gglow.func" operation represents a user defined function. These are
        callable SSA-region operations that contain toy computations.

        Example:

        ```
        gglow.func @main() {
        %0 = gglow.constant dense<5.500000e+00> : tensor<f64>
        %1 = gglow.reshape(%0 : tensor<f64>) -> tensor<2x2xf64>
        gglow.print %1 : tensor<2x2xf64>
        gglow.return
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
        Region *getCallableRegion() { return &getBody(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}
```

Since we have mentioned it has customAssemblyFormat, we will be using custom printer parser which is implemented using some internal functions given by mlir (I am still trying to understand these...).  You can find definition of these functions from [here](https://mlir.llvm.org/doxygen/FunctionInterfaces_8h_source.html).
```c++
void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result)
{
    auto buildFuncType =
        [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string &)
    { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p)
{
    mlir::function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}
```

The only main thing we have to concern ourselves with are the two traits used:
- `FunctionOpInterface`: It provides types, methods and data structures needed to implement function operation. (no documentation found :/)
- `IsolatedFromAbove`: This trait makes the symbols defined within function invisible to the parent regions it is part of. Read more [here](https://mlir.llvm.org/docs/Traits/#isolatedfromabove)

## 2. Return Operation
The Block inside the function operation encapsulates all the computation (via other operations). To keep things simple (like a C++ function), it has one SSA - Control Flow Graph region, where there are multiple blocks. The instruction inside this block is executed in-order and each block has a termination operator which transfers control to other blocks.

Why am I explaining this ? It is because the return operation is our termination operator that transfer control to other block.

```tablegen
def ReturnOp : GGlowOp<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```
            gglow.func @foo() -> tensor<2xf64> {
            ...
            gglow.return %0 : tensor<2xf64>
        }
        ```
    }];

    let arguments = (ins Variadic<F64Tensor>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];
}
```

The only part we have to concern ourselves are the traits used:
- `HasParent<"FuncOp">`: This tells us that the return operation is child operator to `FuncOp` which we defined before and provides API and verifiers needed for such operations. Read more from [here](https://mlir.llvm.org/docs/Traits/#hasparent)
- `Terminator`: This trait provides verification and functionality for operations that are known to be [terminators](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions) .

## 3. Call Operation
This is basically an operation that calls the function defined. Here we define the function to take in only `F64Tensors` as arguments (but this can be made much more extensive)
```tablegen
def GenericCallOp : GGlowOp <"generic_call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to
        be specialized for the shape of its arguments. The callee name is attached
        as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
        %4 = gglow.generic_call @my_func(%1, %3)
            : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
        arguments.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

    let results = (outs F64Tensor);

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];
    
    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];
}
```

## Defining the Inline Implementation using MLIR Interface

We can just use a wrapper provided by MLIR called `DialectInlinerInterface`. This class which implements in-lining feature by deriving it from base-class using `CRTP` (Curiously Recursive Template Pattern), this allows us to make inheritance go backward and allow parent classes to use the instance of derived class members. 
definition:
```c++
class DialectInlinerInterface
: public DialectInterface::Base<DialectInlinerInterface> {
	// body of the class
}
```

Now, we define the following (based of toy chapter 4):

```c++
struct GGlowInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *call, Operation *callable,
                            bool wouldBeCloned) const final {
        return true;
    }

    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
        return true;
    }

    bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
        return true;
    }

    void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
        auto returnOp = cast<ReturnOp>(op);

        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
};
```

I will update the working of handle-terminator later. Later we have to register this interface and then enable the in-lining pass via Pass manager
```c++
void GlowDialect::initialize()
{
    addOperations<
    #define GET_OP_LIST
        #include "lib/Dialect/GGlow/GGlowOps.cpp.inc"
    >();

    addInterface<GGlowInlinerInterface>();

}
```

```c++
mlir::PassManager pm(module.get()->getName());
pm.addPass(mlir::createInlinerPass());
```

# Result

From the above implementation, we can achieve following transformation of IR
```mlir
module {
	gglow.func @transpose_simplify() -> tensor<2x3xf64> {
		%0 = gglow.constant ( dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
			: tensor<2x3xf64> ) -> tensor<2x3xf64>
		%1 = gglow.transpose (%0: tensor<2x3xf64>) -> tensor<3x2xf64>
		%2 = gglow.transpose (%1: tensor<3x2xf64>) -> tensor<2x3xf64>
		gglow.return %2 : tensor<2x3xf64>
	}

	gglow.func @main() {
		%0 = gglow.generic_call @transpose_simplify() : () -> tensor<2x3xf64>
		gglow.print %0 : tensor<2x3xf64>
		gglow.return
	}
}
```
to
```mlir
module {
  gglow.func @transpose_simplify() -> tensor<2x3xf64> {
    %0 = gglow.constant(dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>) -> tensor<2x3xf64>
    gglow.return %0 : tensor<2x3xf64>
  }
  gglow.func @main() {
    %0 = gglow.constant(dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>) -> tensor<2x3xf64>
    gglow.print %0 : tensor<2x3xf64>
    gglow.return
  }
}
```

