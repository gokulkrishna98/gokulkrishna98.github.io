---
title: "OperationPass in MLIR"
date: 2024-08-29
draft: false
ShowToc: true
# cover:
#     image: "/images/gokul_image.jpeg"
#     responsiveImages: false
---

We will be reviewing the shape inference pass implemented in the toy chapter 4. In this article, we will be seeing how to create interface for operation and use that interface to perform modification to IR. The operation which satisfy condition for modification must implement this interface.

# Interface for Operation
We can create interface for an operation by inheriting `OpInterface` class. We can declare the functions that the interface forces the entity to implement can be added via `InterfaceMethod`.

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {

    let description = [{
        Interface to access a registered method to infer the return types for an
        operation that can be used during type inference.
    }];

    let methods = [
        InterfaceMethod<"Infer and set the output shape for the current operation",
            "void", "inferShapes">
    ];
}
```

We can force an Operation to follow this interface by including it as a trait during operation definition. This is achieved by using `DeclareOpInterfaceMethod` and passing the interface to it. for example:

```tablegen
def CastOp : GGlowOp <"cast", [ 
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    Pure, 
    SameOperandsAndResultShape]> {
    ...
}
```

This forces the operation to implement `inferShapes()` method, which writes the output shape of tensor based on arguments given.

# Creating a Operation Pass
Here we create a operation pass that works on operation in `gglow.func`. We iterate through all the operation and check if it has dynamic shapes (unknown) and call the `inferShapes()` to write it. This will make all the shapes known.

We do this by creating a pass by inheriting `mlir::PassWrapper`, it has a virtual method called `runOnOperation`, which executed by `OperationPassManager`. We override this method to implement the above logic.

```c++
struct ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass, OperationPass<mlir::gglow::FuncOp>> {
...
    void runOnOperation() override {
        auto f = getOperation();

        llvm::SmallVector<mlir::Operation*, 16> opWorklist;
        f.walk([&](mlir::Operation* op) {
            if(returnsDynamicShape(op)){
                opWorklist.push_back(op);
            }
        });

        for(size_t i =0; i<opWorklist.size(); i++) {
            Operation *op = opWorklist[i];
            if (auto shapeOp = dyn_cast<ShapeInference>(op)){
                shapeOp.inferShapes();
            } else {
                op->emitError("unable to infer shape of operation without shape "
                            "inference interface");
                return signalPassFailure();
            }
        }

        opWorklist.clear();
        return;
    }
...
};
```

# Creating and Running the Operation via PassManager

We create a pass manager by creating a unique pointer to the Pass.
```c++
std::unique_ptr<mlir::Pass> createShapeInferencePass(){
    return std::make_unique<ShapeInferencePass>();
}
```

Then we create a pass manager and add the Pass:
```c++
mlir::PassManager pm(module.get()->getName());
pm.addPass(mlir::createInlinerPass());

auto &optPM = pm.nest<mlir::gglow::FuncOp>();
optPM.addPass(mlir::gglow::createShapeInferencePass());
optPM.addPass(mlir::createCanonicalizerPass());
```

# Result
This results in following optimization:
Input:
```mlir
gglow.func @transpose_simplify(%arg0 : tensor<*xf64>) -> tensor<*xf64> {
	%0 = gglow.transpose (%arg0: tensor<*xf64>) -> tensor<*xf64>
	%1 = gglow.transpose (%0: tensor<*xf64>) -> tensor<*xf64>
	gglow.return %1 : tensor<*xf64>
}
gglow.func @main() {
	%0 = gglow.constant ( dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
		: tensor<2x3xf64> ) -> tensor<2x3xf64>
	%1 = gglow.generic_call @transpose_simplify(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
	gglow.print %1 : tensor<*xf64>
	gglow.return
}
```

Output after the Pass:
```mlir
  gglow.func @transpose_simplify(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    gglow.print %arg0 : tensor<*xf64>
    gglow.return %arg0 : tensor<*xf64>
  }
  gglow.func @main() {
    %0 = gglow.constant(dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>) -> tensor<2x3xf64>
    gglow.print %0 : tensor<2x3xf64>
    gglow.print %0 : tensor<2x3xf64>
    gglow.return
  }

```