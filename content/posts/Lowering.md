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

## Introduction
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

In our case, we treat high level operations like multiplication, convolution as high level operations and dialect which defines these are high level dialect (the `A` operation in diagram). Later we convert these high level operations to operations from a set of standard dialects in mlir (like scf, affine, memref). These are intermediate operations, the `B` in diagram, the reason we are doing this because we have predefined pattern-rewrites for converting these operations to operations llvm dialect. The llvm-dialect is our low-level dialect. We can use the JIT (defined by MLIR) to execute llvm-dialect in the module. This article will discuss three main parts:
- Converting high level dialect to standard (intermediate) dialect.
- Convert standard dialect to llvm-dialect.
- Set up JIT to execute the llvm-dialect.

Note: Will be using the term standard dialect, instead of intermediate as it makes more sense.
Note: MLIR gives option to make whole dialect as legal or illegal. So, when i mention operations (it includes operations 😀 in this case).
## How to lower a Dialect?
This is a quick overview of how to convert a high-level dialect to a standard one. the general steps remain the same except for creating the new operations, which depend on the standard dialects.
mlir provides a dialect conversion framework, which converts illegal dialects to legal ones. there are three things required:
- conversion target: formal specification of which dialects are legal.
- a set of rewrite patterns.
- (optional) type converter.

### 1. Creating a Pass using Dialect Conversion Framework.
We will be creating a Pass using `PassWrapper` class to implement this lowering. This pass will encompass conversion target, register the rewrite patterns and specify type-converter.

```c++
struct LoweringPass : 
	public PassWrapper<LoweringPass, OperationPass<ModuleOp>> {
	void getDependentDialects(DialectRegistry &registry) const override;
	void runOnOperation() final;
}
```

Lowering Pass is a class created using CRTP (read further [here](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)) which takes in two template class arguments, one is the lowering class we are implementing and another is a Operation Pass specifying which level (operation) the defined lowering performed (most of the cases it is ModuleOp). 

CRTP allows us create interfaces, which is defined by derived class (the class we are implementing) and the API will use this class to perform the required lowering logic using the interfaces we implemented (similar to inheritance with virtual function, CRTP has more strict compile time checking due to templates).

Our work, now boils down to implementing two functions in the current derived class. These functions are:
1. `getDependentDialects(DialectRegistry &registry`) : In this functions we have to registers the dialect. These are the dialect of operations, type etc which we will be creating in this lowering. Ex,  X1 -> (Y1, Y2, Y3), then we have to register Y1, Y2 and Y3 using this function. Example:
```c++
void getDependentDialects(DialectRegistry &registry) const override {
	registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
}
```
2. `runOnOperation()` : The polymorphic API that runs the pass over the currently held operation. This is the function, where we specify which operations are legal or illegal, specify the pattern rewrites and then apply conversion. Example:
```c++
void GGlowToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,  patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  patterns.add<PrintOpLowering>(&getContext());

  auto module = getOperation();
  if(failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
```

### 2. Creating a Pattern Rewrite

Generally, we create pattern rewrite by deriving an object of `Conversion Pattern` class. In ConversionPattern, we first specify a particular operation (pattern), when it encounters this pattern in the IR, it triggers the function `matchAndRewrite`. This function is responsible for converting the high level operation to the lower ones. This involves erasing the pattern operation and then writing the new operations (in dependent dialect), which gives same semantic meaning. We specify the pattern operation in the parent constructor.
Example of simple lowering (one to one mapping), this pattern rewrites the high level `elementwise` operation to `affine` loops + `arith` operations.

```c++
// LOWERING BINARY OPS
template<typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
	BinaryOpLowering(MLIRContext *ctx) : 
	ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {};

LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands
	ConversionPatternRewriter &rewriter) const final {
	auto loc = op->getLoc();
	lowerOpToLoops(op, operands, rewriter, [loc](OpBuilder &builder,
		ValueRange mem_ref_operands,ValueRange ivs){
			typename BinaryOp::Adaptor binaryAdaptor(mem_ref_operands);
			auto loaded_lhs = builder.create<affine::AffineLoadOp>(loc, 
				binaryAdaptor.getLhs(), ivs);

			auto loaded_rhs = builder.create<affine::AffineLoadOp>(loc, 
				binaryAdaptor.getRhs(), ivs);
			
			return builder.create<LoweredBinaryOp>(loc, loaded_lhs,
				loaded_rhs);
	});
	return success();
}

};

using AddOpLowering = BinaryOpLowering<gglow::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<gglow::MulOp, arith::MulFOp>;
```

### 3. TypeConverter (optional)
For complex lowering, where we have to convert types to another one (it can be singular or multiple types), for further reading check [here](https://mlir.llvm.org/docs/DialectConversion/#type-converter). In this context, we will be using types in the standard dialect, so default TypeConverter Object is good enough for us.

## Converting Standard Dialect to LLVM Dialect
In the `GGlowToLLVMLoweringPass::runOnOperation()`, you can add predefined conversion patterns from standard dialects to the LLVM. Now, the pass will lower everything to LLVM Dialect at the end of conversion.

## Set up JIT to run the LLVM Dialect.

The Execution engine provides us a way to execute any functions in our mlir via their JIT codegen.
```c++
int runJit(mlir::ModuleOp module) {
	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	
	
	mlir::registerBuiltinDialectTranslation(*module->getContext());
	mlir::registerLLVMDialectTranslation(*module->getContext());
	
	// An optimization pipeline to use within the execution engine.
	auto optPipeline = mlir::makeOptimizingTransformer(3, 
		/*sizeLevel=*/0,/*targetMachine=*/nullptr);
	
	// Create an MLIR execution engine.
	mlir::ExecutionEngineOptions engineOptions;
	engineOptions.transformer = optPipeline;
	auto maybeEngine = mlir::ExecutionEngine::create(module, 
		engineOptions);
	assert(maybeEngine && "failed to construct an execution engine");
	auto &engine = maybeEngine.get();
	
	// Invoke the JIT-compiled function.
	auto invocationResult = engine->invokePacked("main");
	
	if (invocationResult) {
		llvm::errs() << "JIT invocation failed\n";
		return -1;
	}
	return 0;
}
```