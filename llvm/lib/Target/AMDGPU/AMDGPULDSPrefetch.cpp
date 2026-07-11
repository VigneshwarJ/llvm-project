//===-- AMDGPULDSPrefetch.cpp - Prefetch LDS loads one iteration ahead ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements software pipelining of LDS (Local Data Share) loads
// within loops. It prefetches LDS data one iteration ahead to hide the latency
// between LDS reads and their use by WMMA instructions.
//
// It handles two patterns:
//
// Pattern 1 - Simple linear recurrence:
//   loop:
//     %data = load from LDS[i]
//     use %data (e.g., in WMMA)
//     i++
//
// Pattern 2 - Double-buffering (XOR-based ping-pong):
//   The loop alternates between two LDS buffers using a pattern like:
//     %buf = phi [xor(%buf, 1), loop], [0, preheader]
//     addr = base + stride * %buf
//   For these, we prefetch from the OTHER buffer (xor'd address) which will
//   be ready after the next tensor_load_to_lds completes.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lds-prefetch"

static cl::opt<bool> EnableLDSPrefetch(
    "amdgpu-lds-prefetch", cl::init(false), cl::Hidden,
    cl::desc("Enable LDS prefetching one iteration ahead in loops"));

static cl::opt<unsigned> LDSPrefetchMinLoads(
    "amdgpu-lds-prefetch-min-loads", cl::init(4), cl::Hidden,
    cl::desc("Minimum number of LDS loads in loop to enable prefetching"));

STATISTIC(NumLoopsPrefetched, "Number of loops with LDS prefetching applied");
STATISTIC(NumLoadsPrefetched, "Number of LDS loads prefetched");

namespace {

class AMDGPULDSPrefetch : public FunctionPass {
public:
  static char ID;

  AMDGPULDSPrefetch() : FunctionPass(ID) {
    initializeAMDGPULDSPrefetchPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  StringRef getPassName() const override { return "AMDGPU LDS Prefetch"; }

private:
  bool processLoop(Loop *L, LoopInfo &LI, ScalarEvolution &SE,
                   DominatorTree &DT);
  bool processLoopDoubleBuffer(Loop *L, LoopInfo &LI, ScalarEvolution &SE,
                               DominatorTree &DT);
  bool isLDSLoad(Instruction *I) const;
  bool hasWMMAUser(Instruction *I) const;
  PHINode *findDoubleBufferPHI(Loop *L) const;
  Value *findAlternateBufferIndex(PHINode *BufferPHI) const;
};

} // end anonymous namespace

char AMDGPULDSPrefetch::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPULDSPrefetch, DEBUG_TYPE,
                      "AMDGPU LDS Prefetch", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPULDSPrefetch, DEBUG_TYPE,
                    "AMDGPU LDS Prefetch", false, false)

bool AMDGPULDSPrefetch::isLDSLoad(Instruction *I) const {
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    unsigned AS = LI->getPointerAddressSpace();
    // LDS is address space 3
    return AS == 3;
  }
  // Also check for AMDGPU-specific LDS intrinsics like ds.load.tr16
  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (Function *F = CI->getCalledFunction()) {
      if (F->getIntrinsicID() == Intrinsic::amdgcn_ds_load_tr16_b128)
        return true;
    }
  }
  return false;
}

bool AMDGPULDSPrefetch::hasWMMAUser(Instruction *I) const {
  for (User *U : I->users()) {
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (Function *F = CI->getCalledFunction()) {
        Intrinsic::ID IID = F->getIntrinsicID();
        // Check for WMMA intrinsics
        if (IID == Intrinsic::amdgcn_wmma_f32_16x16x32_bf16 ||
            IID == Intrinsic::amdgcn_wmma_f32_16x16x16_bf16 ||
            IID == Intrinsic::amdgcn_wmma_f16_16x16x16_f16 ||
            IID == Intrinsic::amdgcn_wmma_bf16_16x16x16_bf16 ||
            IID == Intrinsic::amdgcn_wmma_i32_16x16x16_iu8 ||
            IID == Intrinsic::amdgcn_wmma_i32_16x16x16_iu4)
          return true;
      }
    }
    // Recursively check through bitcasts, shufflevector, etc.
    if (auto *BC = dyn_cast<BitCastInst>(U)) {
      if (hasWMMAUser(BC))
        return true;
    }
    if (auto *SV = dyn_cast<ShuffleVectorInst>(U)) {
      if (hasWMMAUser(SV))
        return true;
    }
    if (auto *IE = dyn_cast<InsertElementInst>(U)) {
      if (hasWMMAUser(IE))
        return true;
    }
  }
  return false;
}

bool AMDGPULDSPrefetch::processLoop(Loop *L, LoopInfo &LI, ScalarEvolution &SE,
                                     DominatorTree &DT) {
  // Only process innermost loops
  if (!L->isInnermost())
    return false;

  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();

  if (!Preheader || !Header || !Latch)
    return false;

  // Collect LDS loads that are used by WMMA instructions
  SmallVector<Instruction *, 32> LDSLoads;
  SmallVector<Value *, 32> LoadPtrs;
  SmallVector<bool, 32> IsSimpleAddRec; // True if simple AddRecExpr, false if nested

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (isLDSLoad(&I)) {
        // Check if this load's address is loop-variant
        Value *Ptr = nullptr;
        if (auto *LI = dyn_cast<LoadInst>(&I))
          Ptr = LI->getPointerOperand();
        else if (auto *CI = dyn_cast<CallInst>(&I))
          Ptr = CI->getArgOperand(0);

        if (!Ptr) {
          LLVM_DEBUG(dbgs() << "  Skipping LDS load (no ptr): " << I << "\n");
          continue;
        }

        if (L->isLoopInvariant(Ptr)) {
          LLVM_DEBUG(dbgs() << "  Skipping loop-invariant LDS load: " << I
                            << "\n    Ptr: " << *Ptr << "\n");
          continue;
        }

        // Get SCEV for the pointer
        const SCEV *PtrSCEV = SE.getSCEV(Ptr);
        LLVM_DEBUG(dbgs() << "  LDS load: " << I << "\n    Ptr: " << *Ptr
                          << "\n    SCEV: " << *PtrSCEV << "\n");

        // Check if this is a simple AddRecExpr
        const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrSCEV);
        if (AR && AR->getLoop() == L) {
          // This load has a loop-recurrent address pattern
          LDSLoads.push_back(&I);
          LoadPtrs.push_back(Ptr);
          IsSimpleAddRec.push_back(true);
          LLVM_DEBUG(dbgs() << "Found LDS load with recurrent address: " << I
                            << "\n  SCEV: " << *AR << "\n");
        } else {
          // For now, skip complex patterns - they require special handling
          // (e.g., double-buffering with XOR-based ping-pong)
          LLVM_DEBUG(dbgs() << "    No simple AddRecExpr found, skipping\n");
        }
      }
    }
  }

  // Check if we have enough loads to make prefetching worthwhile
  if (LDSLoads.size() < LDSPrefetchMinLoads) {
    LLVM_DEBUG(dbgs() << "Loop has only " << LDSLoads.size()
                      << " LDS loads, need at least " << LDSPrefetchMinLoads
                      << "\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Processing loop with " << LDSLoads.size()
                    << " LDS loads for prefetching\n");

  // For each LDS load, we need to:
  // 1. Clone the load in the preheader (for initial iteration)
  // 2. Create a PHI to select between preheader load and loop load
  // 3. Advance the address to load the next iteration's data

  bool Changed = false;
  SCEVExpander Expander(SE, "lds-prefetch");

  for (size_t i = 0; i < LDSLoads.size(); ++i) {
    Instruction *Load = LDSLoads[i];
    Value *Ptr = LoadPtrs[i];

    const SCEV *PtrSCEV = SE.getSCEV(Ptr);
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(PtrSCEV);

    // Compute the address for the next iteration: AR + Step
    const SCEV *NextPtrSCEV =
        SE.getAddExpr(AR, AR->getStepRecurrence(SE));

    // Check if we can safely expand the next-iteration address
    if (!Expander.isSafeToExpand(NextPtrSCEV)) {
      LLVM_DEBUG(dbgs() << "Cannot expand next-iteration address for: " << *Load
                        << "\n");
      continue;
    }

    // Get the initial address (start value of the AddRec)
    const SCEV *InitPtrSCEV = AR->getStart();
    if (!Expander.isSafeToExpand(InitPtrSCEV)) {
      LLVM_DEBUG(dbgs() << "Cannot expand initial address for: " << *Load
                        << "\n");
      continue;
    }

    // Create the prefetch load in the preheader
    IRBuilder<> PreheaderBuilder(Preheader->getTerminator());

    // Expand the initial pointer address
    Value *InitPtr =
        Expander.expandCodeFor(InitPtrSCEV, Ptr->getType(),
                               Preheader->getTerminator());

    // Clone the load for the preheader
    Instruction *PrefetchLoad = Load->clone();
    if (auto *LI = dyn_cast<LoadInst>(PrefetchLoad)) {
      LI->setOperand(LI->getPointerOperandIndex(), InitPtr);
    } else if (auto *CI = dyn_cast<CallInst>(PrefetchLoad)) {
      CI->setArgOperand(0, InitPtr);
    }
    PrefetchLoad->setName(Load->getName() + ".prefetch");
    PrefetchLoad->insertBefore(Preheader->getTerminator()->getIterator());

    // Create PHI node at the beginning of the header
    IRBuilder<> HeaderBuilder(&*Header->getFirstInsertionPt());
    PHINode *LoadPHI =
        HeaderBuilder.CreatePHI(Load->getType(), 2, Load->getName() + ".phi");

    // Expand the next-iteration address in the loop body
    // Insert just before the original load
    Value *NextPtr =
        Expander.expandCodeFor(NextPtrSCEV, Ptr->getType(), Load);

    // Clone the load for the next iteration (in-loop prefetch)
    Instruction *NextLoad = Load->clone();
    if (auto *LI = dyn_cast<LoadInst>(NextLoad)) {
      LI->setOperand(LI->getPointerOperandIndex(), NextPtr);
    } else if (auto *CI = dyn_cast<CallInst>(NextLoad)) {
      CI->setArgOperand(0, NextPtr);
    }
    NextLoad->setName(Load->getName() + ".next");
    NextLoad->insertAfter(Load->getIterator());

    // Set up the PHI incoming values
    LoadPHI->addIncoming(PrefetchLoad, Preheader);
    LoadPHI->addIncoming(NextLoad, Latch);

    // Replace uses of the original load with the PHI
    // But we need to be careful: only replace uses that are dominated by the PHI
    // and not the NextLoad itself
    SmallVector<Use *, 16> UsesToReplace;
    for (Use &U : Load->uses()) {
      Instruction *UserInst = cast<Instruction>(U.getUser());
      // Don't replace the use in NextLoad
      if (UserInst == NextLoad)
        continue;
      // Only replace uses in the loop
      if (L->contains(UserInst))
        UsesToReplace.push_back(&U);
    }

    for (Use *U : UsesToReplace)
      U->set(LoadPHI);

    // Remove the original load if it has no more uses
    if (Load->use_empty())
      Load->eraseFromParent();

    ++NumLoadsPrefetched;
    Changed = true;
    LLVM_DEBUG(dbgs() << "Prefetched LDS load: " << *PrefetchLoad << "\n");
  }

  if (Changed)
    ++NumLoopsPrefetched;

  return Changed;
}

// Find a PHI node that follows the double-buffering pattern:
// %phi = phi [xor(%phi, 1), loop], [0, preheader]
PHINode *AMDGPULDSPrefetch::findDoubleBufferPHI(Loop *L) const {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();

  if (!Header || !Preheader || !Latch)
    return nullptr;

  LLVM_DEBUG(dbgs() << "Searching for double-buffer PHI in loop with header "
                    << Header->getName() << ", preheader " << Preheader->getName()
                    << ", latch " << Latch->getName() << "\n");

  for (PHINode &PHI : Header->phis()) {
    // Check if it's an integer PHI
    if (!PHI.getType()->isIntegerTy()) {
      LLVM_DEBUG(dbgs() << "  Skipping non-integer PHI: " << PHI << "\n");
      continue;
    }

    // For single-block loops (header == latch), check all predecessors
    Value *InitVal = nullptr;
    Value *LoopVal = nullptr;

    for (unsigned i = 0; i < PHI.getNumIncomingValues(); ++i) {
      BasicBlock *IncomingBB = PHI.getIncomingBlock(i);
      if (L->contains(IncomingBB)) {
        LoopVal = PHI.getIncomingValue(i);
      } else {
        InitVal = PHI.getIncomingValue(i);
      }
    }

    LLVM_DEBUG(dbgs() << "  Checking PHI: " << PHI << "\n    InitVal: "
                      << (InitVal ? *InitVal : *(Value *)nullptr)
                      << "\n    LoopVal: " << (LoopVal ? *LoopVal : *(Value *)nullptr)
                      << "\n");

    // Check if initial value is 0
    auto *CI = dyn_cast_or_null<ConstantInt>(InitVal);
    if (!CI || !CI->isZero())
      continue;

    // Check if the loop value is xor(phi, 1)
    if (auto *XorInst = dyn_cast_or_null<BinaryOperator>(LoopVal)) {
      if (XorInst->getOpcode() == Instruction::Xor) {
        Value *Op0 = XorInst->getOperand(0);
        Value *Op1 = XorInst->getOperand(1);

        // Check if one operand is the PHI and the other is 1
        if ((Op0 == &PHI && isa<ConstantInt>(Op1) &&
             cast<ConstantInt>(Op1)->isOne()) ||
            (Op1 == &PHI && isa<ConstantInt>(Op0) &&
             cast<ConstantInt>(Op0)->isOne())) {
          LLVM_DEBUG(dbgs() << "Found double-buffer PHI: " << PHI << "\n");
          return &PHI;
        }
      }
    }
  }

  return nullptr;
}

// Given a double-buffer PHI (0/1 alternating), find the XOR'd value
// that represents the OTHER buffer index.
Value *AMDGPULDSPrefetch::findAlternateBufferIndex(PHINode *BufferPHI) const {
  // The XOR value should be: xor(BufferPHI, 1)
  // This is typically the value that goes back into the PHI
  LLVM_DEBUG(dbgs() << "Looking for alternate buffer index for PHI: "
                    << *BufferPHI << "\n");

  for (User *U : BufferPHI->users()) {
    LLVM_DEBUG(dbgs() << "  Checking user: " << *U << "\n");
    if (auto *XorInst = dyn_cast<BinaryOperator>(U)) {
      if (XorInst->getOpcode() == Instruction::Xor) {
        Value *Other = XorInst->getOperand(0) == BufferPHI
                           ? XorInst->getOperand(1)
                           : XorInst->getOperand(0);
        LLVM_DEBUG(dbgs() << "    XOR other operand: " << *Other << "\n");
        if (auto *CI = dyn_cast<ConstantInt>(Other)) {
          if (CI->isOne()) {
            LLVM_DEBUG(dbgs() << "Found alternate buffer index: " << *XorInst
                              << "\n");
            return XorInst;
          }
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  No XOR with 1 found\n");
  return nullptr;
}

// Process a loop that uses double-buffering for LDS accesses.
// The transformation prefetches from the alternate buffer.
bool AMDGPULDSPrefetch::processLoopDoubleBuffer(Loop *L, LoopInfo &LI,
                                                 ScalarEvolution &SE,
                                                 DominatorTree &DT) {
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();

  if (!Preheader || !Header || !Latch)
    return false;

  // Find the double-buffer PHI
  PHINode *BufferPHI = findDoubleBufferPHI(L);
  if (!BufferPHI) {
    LLVM_DEBUG(dbgs() << "No double-buffer PHI found\n");
    return false;
  }

  // Find the XOR'd index (the OTHER buffer)
  Value *AltIndex = findAlternateBufferIndex(BufferPHI);
  if (!AltIndex) {
    LLVM_DEBUG(dbgs() << "Could not find alternate buffer index\n");
    return false;
  }

  // Collect LDS loads that use the BufferPHI (directly or indirectly via GEP)
  SmallVector<Instruction *, 32> LDSLoads;
  SmallVector<GetElementPtrInst *, 32> BufferGEPs;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (!isLDSLoad(&I))
        continue;

      Value *Ptr = nullptr;
      if (auto *LI = dyn_cast<LoadInst>(&I))
        Ptr = LI->getPointerOperand();
      else if (auto *CI = dyn_cast<CallInst>(&I))
        Ptr = CI->getArgOperand(0);

      if (!Ptr)
        continue;

      // Check if this pointer uses BufferPHI (through the multiply pattern)
      // The pattern is: GEP(base, stride * BufferPHI) or similar
      // We look for uses of BufferPHI that eventually reach this pointer
      bool UsesBufferPHI = false;
      SmallVector<Value *, 8> Worklist;
      SmallPtrSet<Value *, 8> Visited;
      Worklist.push_back(Ptr);

      while (!Worklist.empty()) {
        Value *V = Worklist.pop_back_val();
        if (!Visited.insert(V).second)
          continue;

        if (V == BufferPHI) {
          UsesBufferPHI = true;
          break;
        }

        if (auto *Inst = dyn_cast<Instruction>(V)) {
          for (Use &U : Inst->operands()) {
            if (auto *OpInst = dyn_cast<Instruction>(U.get())) {
              if (L->contains(OpInst))
                Worklist.push_back(OpInst);
            }
          }
        }
      }

      if (UsesBufferPHI) {
        LDSLoads.push_back(&I);
        LLVM_DEBUG(dbgs() << "Found LDS load using buffer PHI: " << I << "\n");
      }
    }
  }

  if (LDSLoads.size() < LDSPrefetchMinLoads) {
    LLVM_DEBUG(dbgs() << "Only " << LDSLoads.size()
                      << " double-buffer LDS loads found\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Processing " << LDSLoads.size()
                    << " double-buffer LDS loads for prefetching\n");

  // The transformation converts:
  //   loop:
  //     buf = phi [xor(buf, 1), loop], [0, preheader]
  //     ; ... barrier ...
  //     addr = base + stride * buf + offset  ; based on current buffer
  //     data = load addr
  //     use data in WMMA
  //
  // Into:
  //   preheader:
  //     ; Load initial data from buffer 0 (which was pre-filled)
  //     init_addr = base + stride * 0 + offset
  //     init_data = load init_addr
  //   loop:
  //     buf = phi [xor(buf, 1), loop], [0, preheader]
  //     data = phi [init_data, preheader], [prefetch_data, loop]
  //     ; ... barrier fills xor(buf,1) buffer ...
  //     ; Prefetch from the ALTERNATE buffer (just filled by barrier)
  //     alt_addr = base + stride * xor(buf, 1) + offset
  //     prefetch_data = load alt_addr  ; for NEXT iteration
  //     use data (from PHI) in WMMA  ; uses CURRENT iteration data

  bool Changed = false;

  // Find the MUL that computes: stride * BufferPHI (e.g., 16640 * BufferPHI)
  Instruction *OrigMul = nullptr;
  for (User *U : BufferPHI->users()) {
    if (auto *Mul = dyn_cast<BinaryOperator>(U)) {
      if (Mul->getOpcode() == Instruction::Mul) {
        OrigMul = Mul;
        break;
      }
    }
  }

  if (!OrigMul) {
    LLVM_DEBUG(dbgs() << "Could not find MUL for buffer PHI\n");
    return false;
  }

  // Find the corresponding MUL that uses AltIndex (stride * AltIndex)
  Instruction *AltMul = nullptr;
  Value *Stride = (OrigMul->getOperand(0) == BufferPHI)
                      ? OrigMul->getOperand(1)
                      : OrigMul->getOperand(0);

  for (User *U : AltIndex->users()) {
    if (auto *Mul = dyn_cast<BinaryOperator>(U)) {
      if (Mul->getOpcode() == Instruction::Mul) {
        if (Mul->getOperand(0) == Stride || Mul->getOperand(1) == Stride) {
          AltMul = Mul;
          break;
        }
      }
    }
  }

  if (!AltMul) {
    LLVM_DEBUG(dbgs() << "Could not find alternate MUL (stride * AltIndex)\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Found OrigMul: " << *OrigMul << "\n  AltMul: " << *AltMul
                    << "\n  Stride: " << *Stride << "\n");

  // For each LDS load, transform it
  for (Instruction *Load : LDSLoads) {
    Value *Ptr = nullptr;
    if (auto *LI = dyn_cast<LoadInst>(Load))
      Ptr = LI->getPointerOperand();
    else if (auto *CI = dyn_cast<CallInst>(Load))
      Ptr = CI->getArgOperand(0);

    if (!Ptr)
      continue;

    // Find the chain of instructions from OrigMul to Ptr
    // First, collect all instructions reachable from Ptr (going backwards through operands)
    SmallPtrSet<Instruction *, 16> Reachable;
    SmallVector<Value *, 16> Worklist;

    Worklist.push_back(Ptr);
    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      auto *Inst = dyn_cast<Instruction>(V);
      if (!Inst)
        continue;
      if (!Reachable.insert(Inst).second)
        continue;
      // Don't traverse past OrigMul - it's our substitution boundary
      if (Inst == OrigMul)
        continue;
      for (Value *Op : Inst->operands())
        Worklist.push_back(Op);
    }

    // Now find which of these instructions actually depend on OrigMul
    // An instruction depends on OrigMul if it uses OrigMul or uses something that depends on OrigMul
    SmallPtrSet<Instruction *, 16> DependsOnOrigMul;
    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (Instruction *Inst : Reachable) {
        if (DependsOnOrigMul.count(Inst))
          continue;
        for (Value *Op : Inst->operands()) {
          if (Op == OrigMul || DependsOnOrigMul.count(dyn_cast<Instruction>(Op))) {
            DependsOnOrigMul.insert(Inst);
            Changed = true;
            break;
          }
        }
      }
    }

    // Collect instructions to clone (those that depend on OrigMul and are reachable from Ptr)
    SmallVector<Instruction *, 8> ToClone;
    for (Instruction *Inst : Reachable) {
      if (DependsOnOrigMul.count(Inst))
        ToClone.push_back(Inst);
    }

    // If Ptr doesn't depend on OrigMul at all, skip
    if (ToClone.empty()) {
      LLVM_DEBUG(dbgs() << "  Load ptr doesn't depend on OrigMul, skipping: " << *Load << "\n");
      continue;
    }

    // Sort ToClone topologically: dependencies before dependents
    SmallVector<Instruction *, 8> Sorted;
    SmallPtrSet<Instruction *, 8> Added;

    std::function<void(Instruction *)> addWithDeps = [&](Instruction *I) {
      if (!DependsOnOrigMul.count(I) || Added.count(I))
        return;
      // First add all dependencies
      for (Value *Op : I->operands()) {
        if (auto *OpInst = dyn_cast<Instruction>(Op))
          addWithDeps(OpInst);
      }
      Sorted.push_back(I);
      Added.insert(I);
    };

    for (Instruction *I : ToClone)
      addWithDeps(I);

    ToClone = std::move(Sorted);

    LLVM_DEBUG(dbgs() << "  Cloning " << ToClone.size() << " instructions for load: " << *Load << "\n");

    // Find insertion point for prefetch loads
    // We need to place the prefetch load at the END of the loop body (before the
    // terminator) so that:
    // 1. It's defined after the PHI nodes at the start of the block
    // 2. It's defined before the back-edge is taken
    // This ensures the PHI can reference the prefetch load from the previous iteration
    Instruction *InsertPt = Latch->getTerminator();

    // Clone for alternate buffer (prefetch)
    DenseMap<Value *, Value *> VMap;
    VMap[OrigMul] = AltMul;

    for (Instruction *I : ToClone) {
      if (VMap.count(I))
        continue;
      Instruction *Clone = I->clone();
      Clone->setName(I->getName() + ".alt");
      Clone->insertBefore(InsertPt->getIterator());
      // Remap operands
      for (Use &U : Clone->operands()) {
        auto It = VMap.find(U.get());
        if (It != VMap.end())
          U.set(It->second);
      }
      VMap[I] = Clone;
    }

    // Get the cloned pointer
    Value *AltPtr = Ptr;
    if (auto It = VMap.find(Ptr); It != VMap.end())
      AltPtr = It->second;

    // Clone the load with the alternate pointer (prefetch load)
    Instruction *PrefetchLoad = Load->clone();
    if (auto *LI = dyn_cast<LoadInst>(PrefetchLoad))
      LI->setOperand(LI->getPointerOperandIndex(), AltPtr);
    else if (auto *CI = dyn_cast<CallInst>(PrefetchLoad))
      CI->setArgOperand(0, AltPtr);
    PrefetchLoad->setName(Load->getName() + ".prefetch");
    PrefetchLoad->insertBefore(InsertPt->getIterator());

    // Create initial load in preheader (with BufferPHI = 0)
    // The initial address is: base + stride * 0 + offset = base + offset
    DenseMap<Value *, Value *> InitVMap;
    InitVMap[OrigMul] = ConstantInt::get(OrigMul->getType(), 0);

    for (Instruction *I : ToClone) {
      if (InitVMap.count(I))
        continue;
      Instruction *Clone = I->clone();
      Clone->setName(I->getName() + ".init");
      Clone->insertBefore(Preheader->getTerminator()->getIterator());
      for (Use &U : Clone->operands()) {
        auto It = InitVMap.find(U.get());
        if (It != InitVMap.end())
          U.set(It->second);
      }
      InitVMap[I] = Clone;
    }

    Value *InitPtr = Ptr;
    if (auto It = InitVMap.find(Ptr); It != InitVMap.end())
      InitPtr = It->second;

    Instruction *InitLoad = Load->clone();
    if (auto *LI = dyn_cast<LoadInst>(InitLoad))
      LI->setOperand(LI->getPointerOperandIndex(), InitPtr);
    else if (auto *CI = dyn_cast<CallInst>(InitLoad))
      CI->setArgOperand(0, InitPtr);
    InitLoad->setName(Load->getName() + ".init");
    InitLoad->insertBefore(Preheader->getTerminator()->getIterator());

    // Create PHI node in the header
    // The PHI must have one incoming value per predecessor of the header
    // Preheader -> InitLoad
    // Other predecessors (latches) -> PrefetchLoad
    PHINode *DataPHI = PHINode::Create(Load->getType(), 2, Load->getName() + ".phi");
    DataPHI->insertBefore(Header->begin());
    DataPHI->addIncoming(InitLoad, Preheader);

    // For each predecessor of Header that is in the loop (back edges), use PrefetchLoad
    for (BasicBlock *Pred : predecessors(Header)) {
      if (Pred != Preheader && L->contains(Pred)) {
        DataPHI->addIncoming(PrefetchLoad, Pred);
      }
    }

    // Replace uses of the original load with the PHI
    // Collect uses first to avoid iterator invalidation
    SmallVector<Use *, 16> UsesToReplace;
    for (Use &U : Load->uses()) {
      auto *UserInst = dyn_cast<Instruction>(U.getUser());
      if (!UserInst)
        continue;
      // Don't replace the prefetch load's pointer (it uses the alt address, not Load itself)
      if (UserInst == PrefetchLoad)
        continue;
      // Don't replace uses in preheader
      if (UserInst->getParent() == Preheader)
        continue;
      // Only replace uses within the loop
      if (L->contains(UserInst))
        UsesToReplace.push_back(&U);
    }

    for (Use *U : UsesToReplace)
      U->set(DataPHI);

    // Don't remove original load yet - it might be used by something outside the loop
    // Let DCE clean it up if truly unused
    if (Load->use_empty()) {
      Load->eraseFromParent();
    }

    ++NumLoadsPrefetched;
    Changed = true;
    LLVM_DEBUG(dbgs() << "Prefetched double-buffer load:\n"
                      << "  Init: " << *InitLoad << "\n"
                      << "  Prefetch: " << *PrefetchLoad << "\n"
                      << "  PHI: " << *DataPHI << "\n");
  }

  if (Changed)
    ++NumLoopsPrefetched;

  return Changed;
}

bool AMDGPULDSPrefetch::runOnFunction(Function &F) {
  if (!EnableLDSPrefetch)
    return false;

  // Only run on AMDGPU kernels
  if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return false;

  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  bool Changed = false;

  // Process all loops, innermost first
  SmallVector<Loop *, 8> Worklist;
  for (Loop *L : LI)
    for (Loop *InnerL : depth_first(L))
      Worklist.push_back(InnerL);

  for (Loop *L : Worklist) {
    // First try simple linear recurrence pattern
    bool ProcessedSimple = processLoop(L, LI, SE, DT);
    Changed |= ProcessedSimple;

    // If simple pattern didn't work, try double-buffering pattern
    if (!ProcessedSimple)
      Changed |= processLoopDoubleBuffer(L, LI, SE, DT);
  }

  return Changed;
}

char &llvm::AMDGPULDSPrefetchID = AMDGPULDSPrefetch::ID;

FunctionPass *llvm::createAMDGPULDSPrefetchPass() {
  return new AMDGPULDSPrefetch();
}
