//===- SISGPRCopyOpt.cpp - Hoist SGPR->VGPR copies out of loops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass optimizes SGPR to VGPR copies that appear at loop headers as a
/// result of PHI elimination. When a VGPR PHI has an SGPR initial value, the
/// PHI elimination pass inserts a V_MOV from SGPR to VGPR at the beginning of
/// the loop body. If this SGPR value is loop-invariant, we can hoist the MOV
/// to the loop preheader, avoiding the copy on every iteration.
///
/// This pass runs after register allocation and operates on physical registers.
///
/// Pattern detected:
///   bb.preheader:
///     ; empty or other code
///     S_BRANCH bb.loop
///
///   bb.loop:
///     $vgpr = V_MOV_B64_e32 $sgpr  ; loop-invariant!
///     ...
///     $vgpr = WMMA ...  ; $vgpr redefined by loop body
///     ...
///     S_CBRANCH bb.loop
///
/// Optimization:
///   bb.preheader:
///     $vgpr = V_MOV_B64_e32 $sgpr  ; hoisted!
///     S_BRANCH bb.loop
///
///   bb.loop:
///     ; MOV removed
///     ...
///     $vgpr = WMMA ...
///     ...
///     S_CBRANCH bb.loop
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-sgpr-copy-opt"

static cl::opt<bool> EnableSGPRCopyOpt(
    "amdgpu-enable-sgpr-copy-opt",
    cl::desc("Hoist loop-invariant SGPR->VGPR copies to loop preheaders"),
    cl::init(true));

namespace {

class SISGPRCopyOpt {
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  MachineLoopInfo *MLI;
  MachineFunction *MF;

public:
  SISGPRCopyOpt(MachineLoopInfo *MLI) : MLI(MLI) {}

  bool run(MachineFunction &MF);

private:
  /// Check if a register is an SGPR and is loop-invariant (not defined in the
  /// loop).
  bool isLoopInvariantSGPR(Register Reg, MachineLoop *Loop) const;

  /// Check if a V_MOV instruction can be hoisted to the preheader.
  bool canHoistMov(MachineInstr &MI, MachineLoop *Loop) const;

  /// Hoist eligible V_MOV instructions from loop header to preheader.
  bool hoistLoopInvariantCopies(MachineLoop *Loop);
};

class SISGPRCopyOptLegacy : public MachineFunctionPass {
public:
  static char ID;

  SISGPRCopyOptLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineLoopInfo *MLI =
        &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    SISGPRCopyOpt Impl(MLI);
    return Impl.run(MF);
  }

  StringRef getPassName() const override {
    return "SI SGPR Copy Optimization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SISGPRCopyOptLegacy, DEBUG_TYPE,
                      "SI SGPR Copy Optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SISGPRCopyOptLegacy, DEBUG_TYPE,
                    "SI SGPR Copy Optimization", false, false)

char SISGPRCopyOptLegacy::ID = 0;

char &llvm::SISGPRCopyOptLegacyID = SISGPRCopyOptLegacy::ID;

FunctionPass *llvm::createSISGPRCopyOptLegacyPass() {
  return new SISGPRCopyOptLegacy();
}

bool SISGPRCopyOpt::isLoopInvariantSGPR(Register Reg, MachineLoop *Loop) const {
  if (!Reg.isPhysical())
    return false;

  // Check if this is an SGPR
  if (!TRI->isSGPRPhysReg(Reg))
    return false;

  // Check that the register is not defined anywhere in the loop
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (MachineInstr &MI : *MBB) {
      for (const MachineOperand &MO : MI.defs()) {
        if (MO.isReg() && MO.getReg().isPhysical() &&
            TRI->regsOverlap(MO.getReg(), Reg)) {
          return false;
        }
      }
    }
  }

  return true;
}

bool SISGPRCopyOpt::canHoistMov(MachineInstr &MI, MachineLoop *Loop) const {
  unsigned Opc = MI.getOpcode();

  // Check for V_MOV_B32 or V_MOV_B64 from SGPR
  if (Opc != AMDGPU::V_MOV_B32_e32 && Opc != AMDGPU::V_MOV_B64_e32)
    return false;

  const MachineOperand &DstOp = MI.getOperand(0);
  const MachineOperand &SrcOp = MI.getOperand(1);

  if (!DstOp.isReg() || !SrcOp.isReg())
    return false;

  Register DstReg = DstOp.getReg();
  Register SrcReg = SrcOp.getReg();

  if (!DstReg.isPhysical() || !SrcReg.isPhysical())
    return false;

  // Source must be a loop-invariant SGPR
  if (!isLoopInvariantSGPR(SrcReg, Loop))
    return false;

  // Destination must be a VGPR
  if (!TRI->isVGPR(*MRI, DstReg))
    return false;

  // Check that the destination VGPR is not used (as a true use, not an
  // implicit-def) before this instruction in the loop header. If it were used
  // before being defined here, that would mean the value from the previous
  // iteration is needed, making hoisting incorrect.
  MachineBasicBlock *HeaderBB = Loop->getHeader();
  for (auto I = HeaderBB->begin(); &*I != &MI; ++I) {
    for (const MachineOperand &MO : I->operands()) {
      // Skip non-register operands
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      // Skip definitions - we only care about uses
      if (MO.isDef())
        continue;
      // Skip implicit operands - they're often implicit-uses for super-register
      // tracking, not real data dependencies
      if (MO.isImplicit())
        continue;
      // Check for true use of our destination register
      if (TRI->regsOverlap(MO.getReg(), DstReg))
        return false;
    }
  }

  return true;
}

bool SISGPRCopyOpt::hoistLoopInvariantCopies(MachineLoop *Loop) {
  MachineBasicBlock *Header = Loop->getHeader();
  MachineBasicBlock *Preheader = Loop->getLoopPreheader();

  if (!Preheader)
    return false;

  // Find the insertion point in the preheader (before the terminator)
  MachineBasicBlock::iterator InsertPt = Preheader->getFirstTerminator();

  bool Changed = false;
  SmallVector<MachineInstr *, 16> ToHoist;

  // Collect movs to hoist - only look at the beginning of the header
  // (before any other non-copy/non-mov instructions)
  for (MachineInstr &MI : *Header) {
    // Stop when we hit a non-mov/non-copy/non-fence instruction
    unsigned Opc = MI.getOpcode();
    if (Opc != AMDGPU::V_MOV_B32_e32 && Opc != AMDGPU::V_MOV_B64_e32 &&
        Opc != AMDGPU::COPY && !MI.isMetaInstruction() &&
        Opc != AMDGPU::ATOMIC_FENCE && Opc != AMDGPU::WAVE_BARRIER)
      break;

    if ((Opc == AMDGPU::V_MOV_B64_e32 || Opc == AMDGPU::V_MOV_B32_e32) &&
        canHoistMov(MI, Loop))
      ToHoist.push_back(&MI);
  }

  for (MachineInstr *MI : ToHoist) {
    LLVM_DEBUG(dbgs() << "Hoisting to preheader: " << *MI);
    MI->removeFromParent();
    Preheader->insert(InsertPt, MI);
    Changed = true;
  }

  return Changed;
}

bool SISGPRCopyOpt::run(MachineFunction &MF_) {
  if (!EnableSGPRCopyOpt)
    return false;

  MF = &MF_;
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MRI = &MF->getRegInfo();

  bool Changed = false;

  // Process all loops, innermost first
  for (MachineLoop *L : MLI->getLoopsInPreorder()) {
    Changed |= hoistLoopInvariantCopies(L);
  }

  return Changed;
}
