//===- SILoopCarriedCopyOpt.cpp - Optimize loop-carried VGPR copies ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass optimizes VGPR copies that arise from PHI elimination in loops
/// with software pipelining (e.g., LDS prefetching). It handles two patterns:
///
/// Pattern 1 - Loop-carried DS loads:
/// When a load instruction's result is copied to a different register at the
/// end of a loop iteration, this pass rewrites the load to write directly to
/// the target register, eliminating the copy.
///
///   bb.loop:
///     ; ... uses of $vgpr0-$vgpr3 (from previous iteration) ...
///     $vgpr4-$vgpr7 = DS_READ_B128 ...  ; load for next iteration
///     ; ... more loop body ...
///     $vgpr0-$vgpr3 = COPY $vgpr4-$vgpr7  ; pass to next iteration
///     S_CBRANCH bb.loop
///
/// Optimization:
///   bb.loop:
///     ; ... uses of $vgpr0-$vgpr3 ...
///     $vgpr0-$vgpr3 = DS_READ_B128 ...  ; load directly to target
///     ; ... more loop body (rewrite uses of $vgpr4-$vgpr7 -> $vgpr0-$vgpr3) ...
///     ; COPY removed
///     S_CBRANCH bb.loop
///
/// Pattern 2 - Redundant SGPR-to-VGPR copies:
/// When a loop-invariant SGPR value is copied to VGPRs multiple times within
/// a loop iteration (e.g., due to register pressure from prefetching), this
/// pass eliminates redundant copies by reusing the VGPR destination.
///
///   bb.loop:
///     $vgpr0-$vgpr7 = V_MOV_B64 $sgpr0-$sgpr7  ; first copy
///     use $vgpr0-$vgpr7
///     ; ... VGPRs get clobbered by other operations ...
///     $vgpr0-$vgpr7 = V_MOV_B64 $sgpr0-$sgpr7  ; redundant copy (same source!)
///     use $vgpr0-$vgpr7
///     S_CBRANCH bb.loop
///
/// Optimization: If the SGPR source is loop-invariant and multiple copies
/// target the same destination, we keep only the first copy and rewrite
/// subsequent uses.
///
/// This pass runs after register allocation and operates on physical registers.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-loop-carried-copy-opt"

static cl::opt<bool> EnableLoopCarriedCopyOpt(
    "amdgpu-enable-loop-carried-copy-opt",
    cl::desc("Eliminate loop-carried VGPR copies by rewriting loads"),
    cl::init(true));

static cl::opt<bool> EnableSGPRCopyElim(
    "amdgpu-enable-sgpr-copy-elim",
    cl::desc("Eliminate redundant SGPR-to-VGPR copies in loops"),
    cl::init(true));

namespace {

class SILoopCarriedCopyOpt {
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  MachineLoopInfo *MLI;
  MachineFunction *MF;

public:
  SILoopCarriedCopyOpt(MachineLoopInfo *MLI) : MLI(MLI) {}

  bool run(MachineFunction &MF);

private:
  /// Check if Reg is used (read) between Start (exclusive) and End (exclusive)
  bool isRegUsedBetween(MachineInstr *Start, MachineInstr *End,
                        MCRegister Reg) const;

  /// Check if Reg is defined between Start (exclusive) and End (exclusive)
  bool isRegDefinedBetween(MachineInstr *Start, MachineInstr *End,
                           MCRegister Reg) const;

  /// Try to eliminate a COPY/move at the end of the loop by rewriting the source
  /// instruction to write directly to the COPY destination.
  bool tryEliminateCopy(MachineInstr &CopyMI, MachineLoop *Loop);

  /// Check if this is a VGPR-to-VGPR move instruction
  bool isVGPRMove(const MachineInstr &MI) const;

  /// Rewrite uses of OldReg to NewReg between Start and End (exclusive)
  void rewriteUsesBetween(MachineInstr *Start, MachineInstr *End,
                          MCRegister OldReg, MCRegister NewReg);

  /// Check if this is a DS load instruction that we can rewrite
  bool isDSLoad(const MachineInstr &MI) const;

  /// Check if this is an SGPR-to-VGPR move instruction
  bool isSGPRToVGPRMove(const MachineInstr &MI) const;

  /// Eliminate redundant SGPR-to-VGPR copies in a loop
  bool eliminateRedundantSGPRCopies(MachineLoop *Loop);

  /// Process all loops
  bool processLoop(MachineLoop *Loop);
};

class SILoopCarriedCopyOptLegacy : public MachineFunctionPass {
public:
  static char ID;

  SILoopCarriedCopyOptLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineLoopInfo *MLI =
        &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    SILoopCarriedCopyOpt Impl(MLI);
    return Impl.run(MF);
  }

  StringRef getPassName() const override {
    return "SI Loop Carried Copy Optimization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SILoopCarriedCopyOptLegacy, DEBUG_TYPE,
                      "SI Loop Carried Copy Optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SILoopCarriedCopyOptLegacy, DEBUG_TYPE,
                    "SI Loop Carried Copy Optimization", false, false)

char SILoopCarriedCopyOptLegacy::ID = 0;

char &llvm::SILoopCarriedCopyOptLegacyID = SILoopCarriedCopyOptLegacy::ID;

FunctionPass *llvm::createSILoopCarriedCopyOptLegacyPass() {
  return new SILoopCarriedCopyOptLegacy();
}

bool SILoopCarriedCopyOpt::isDSLoad(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  // Check for DS read instructions
  switch (Opc) {
  case AMDGPU::DS_READ_B32:
  case AMDGPU::DS_READ_B32_gfx9:
  case AMDGPU::DS_READ_B64:
  case AMDGPU::DS_READ_B64_gfx9:
  case AMDGPU::DS_READ_B128:
  case AMDGPU::DS_READ_B128_gfx9:
  case AMDGPU::DS_READ2_B32:
  case AMDGPU::DS_READ2_B32_gfx9:
  case AMDGPU::DS_READ2_B64:
  case AMDGPU::DS_READ2_B64_gfx9:
    return true;
  default:
    return false;
  }
}

bool SILoopCarriedCopyOpt::isRegUsedBetween(MachineInstr *Start,
                                            MachineInstr *End,
                                            MCRegister Reg) const {
  assert(End->getParent() == Start->getParent() &&
         "Start and End must be in same block");

  for (auto I = std::next(Start->getIterator()); &*I != End; ++I) {
    for (const MachineOperand &MO : I->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      if (MO.isUse() && TRI->regsOverlap(MO.getReg(), Reg))
        return true;
    }
  }
  return false;
}

bool SILoopCarriedCopyOpt::isRegDefinedBetween(MachineInstr *Start,
                                               MachineInstr *End,
                                               MCRegister Reg) const {
  MachineBasicBlock *MBB = Start->getParent();
  assert(End->getParent() == MBB && "Start and End must be in same block");

  for (auto I = std::next(Start->getIterator()); &*I != End; ++I) {
    for (const MachineOperand &MO : I->defs()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      if (TRI->regsOverlap(MO.getReg(), Reg))
        return true;
    }
  }
  return false;
}

void SILoopCarriedCopyOpt::rewriteUsesBetween(MachineInstr *Start,
                                              MachineInstr *End,
                                              MCRegister OldReg,
                                              MCRegister NewReg) {
  for (auto I = std::next(Start->getIterator()); &*I != End; ++I) {
    for (MachineOperand &MO : I->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      if (MO.getReg() == OldReg) {
        MO.setReg(NewReg);
      }
    }
  }
}

bool SILoopCarriedCopyOpt::tryEliminateCopy(MachineInstr &CopyMI,
                                            MachineLoop *Loop) {
  // The instruction must be a VGPR move
  if (!isVGPRMove(CopyMI))
    return false;

  const MachineOperand &DstOp = CopyMI.getOperand(0);
  const MachineOperand &SrcOp = CopyMI.getOperand(1);

  if (!DstOp.isReg() || !SrcOp.isReg())
    return false;

  Register DstReg = DstOp.getReg();
  Register SrcReg = SrcOp.getReg();

  if (!DstReg.isPhysical() || !SrcReg.isPhysical())
    return false;

  // Both must be VGPRs
  if (!TRI->isVGPR(*MRI, DstReg) || !TRI->isVGPR(*MRI, SrcReg))
    return false;

  // Find the instruction that defines SrcReg
  MachineInstr *DefMI = nullptr;
  MachineBasicBlock *LoopBB = CopyMI.getParent();

  // Search backwards from COPY to find the definition of SrcReg
  for (auto I = CopyMI.getReverseIterator(); I != LoopBB->rend(); ++I) {
    if (&*I == &CopyMI)
      continue;

    for (const MachineOperand &MO : I->defs()) {
      if (MO.isReg() && MO.getReg().isPhysical() &&
          TRI->regsOverlap(MO.getReg(), SrcReg)) {
        DefMI = &*I;
        break;
      }
    }
    if (DefMI)
      break;
  }

  if (!DefMI) {
    LLVM_DEBUG(dbgs() << "  No def found for COPY source\n");
    return false;
  }

  // The definition must be a DS load
  if (!isDSLoad(*DefMI)) {
    LLVM_DEBUG(dbgs() << "  Def is not a DS load: " << *DefMI);
    return false;
  }

  // Check that DstReg is not used between the start of the loop and the DefMI
  // This is critical: the DstReg holds the value from the previous iteration
  // and must not be overwritten until after all its uses in this iteration.
  for (auto I = LoopBB->begin(); &*I != DefMI; ++I) {
    for (const MachineOperand &MO : I->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      if (MO.isUse() && TRI->regsOverlap(MO.getReg(), DstReg)) {
        LLVM_DEBUG(dbgs() << "  DstReg is used before def: " << *I);
        return false;
      }
    }
  }

  // Check that DstReg is not defined between DefMI and CopyMI (except by CopyMI)
  if (isRegDefinedBetween(DefMI, &CopyMI, DstReg)) {
    LLVM_DEBUG(dbgs() << "  DstReg is defined between def and copy\n");
    return false;
  }

  // Check that DstReg is not used between DefMI and CopyMI
  // (This is generally safe because the COPY is passing the value forward)
  if (isRegUsedBetween(DefMI, &CopyMI, DstReg)) {
    LLVM_DEBUG(dbgs() << "  DstReg is used between def and copy\n");
    return false;
  }

  // Get the def operand of the DS load
  MachineOperand *LoadDefOp = nullptr;
  for (MachineOperand &MO : DefMI->defs()) {
    if (MO.isReg() && TRI->regsOverlap(MO.getReg(), SrcReg)) {
      LoadDefOp = &MO;
      break;
    }
  }

  if (!LoadDefOp) {
    LLVM_DEBUG(dbgs() << "  Could not find load def operand\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Rewriting DS load to use DstReg directly\n");
  LLVM_DEBUG(dbgs() << "    Load: " << *DefMI);
  LLVM_DEBUG(dbgs() << "    Copy: " << CopyMI);

  // Rewrite uses of SrcReg between DefMI and CopyMI to use DstReg
  rewriteUsesBetween(DefMI, &CopyMI, SrcReg, DstReg);

  // Change the load to write directly to DstReg
  LoadDefOp->setReg(DstReg);

  // Remove the COPY
  CopyMI.eraseFromParent();

  return true;
}

/// Check if this is a VGPR-to-VGPR move instruction
bool SILoopCarriedCopyOpt::isVGPRMove(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  // Check for COPY pseudo or actual V_MOV instructions
  if (MI.isCopy())
    return true;

  switch (Opc) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B64_e32:
  case AMDGPU::V_MOV_B64_PSEUDO:
    return true;
  default:
    return false;
  }
}

bool SILoopCarriedCopyOpt::isSGPRToVGPRMove(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  switch (Opc) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B64_e32:
  case AMDGPU::V_MOV_B64_PSEUDO:
    break;
  default:
    if (!MI.isCopy())
      return false;
    break;
  }

  if (MI.getNumOperands() < 2)
    return false;

  const MachineOperand &DstOp = MI.getOperand(0);
  const MachineOperand &SrcOp = MI.getOperand(1);

  if (!DstOp.isReg() || !SrcOp.isReg())
    return false;

  Register DstReg = DstOp.getReg();
  Register SrcReg = SrcOp.getReg();

  if (!DstReg.isPhysical() || !SrcReg.isPhysical())
    return false;

  // Check if source is SGPR and dest is VGPR
  return TRI->isSGPRReg(*MRI, SrcReg) && TRI->isVGPR(*MRI, DstReg);
}

/// Key for tracking SGPR copies: source register + destination register
struct SGPRCopyKey {
  unsigned SrcReg;
  unsigned DstReg;

  bool operator==(const SGPRCopyKey &Other) const {
    return SrcReg == Other.SrcReg && DstReg == Other.DstReg;
  }
};

namespace llvm {
template <> struct DenseMapInfo<SGPRCopyKey> {
  static inline SGPRCopyKey getEmptyKey() {
    return SGPRCopyKey{~0U, ~0U};
  }
  static inline SGPRCopyKey getTombstoneKey() {
    return SGPRCopyKey{~0U - 1, ~0U - 1};
  }
  static unsigned getHashValue(const SGPRCopyKey &Key) {
    return hash_combine(Key.SrcReg, Key.DstReg);
  }
  static bool isEqual(const SGPRCopyKey &LHS, const SGPRCopyKey &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

/// Check if the VGPR destination of a copy is overwritten by subsequent
/// instructions before any use of the original copy's value.
/// This detects dead SGPR-to-VGPR copies.
/// We look ahead up to LookAhead instructions for the overwrite pattern.
bool isOverwrittenBeforeUse(MachineInstr *CopyMI, const SIRegisterInfo *TRI,
                            unsigned LookAhead = 16) {
  // Get the destination register (which may be a super-register)
  Register DstReg = CopyMI->getOperand(0).getReg();
  MachineBasicBlock *MBB = CopyMI->getParent();
  const MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();

  // Get the size of the destination register in 32-bit units
  unsigned DstSizeInBits = TRI->getRegSizeInBits(DstReg, MRI);
  unsigned DstSize = DstSizeInBits / 32;
  if (DstSize == 0 || DstSize > 32)
    return false;

  // Collect all 32-bit sub-registers of DstReg
  // For vgpr696_vgpr697, these would be vgpr696 and vgpr697
  SmallVector<MCRegister, 8> SubRegs;
  MCRegister BaseReg = DstReg.asMCReg();

  // Get all sub-registers that cover DstReg
  for (MCRegister SubReg : TRI->subregs_inclusive(BaseReg)) {
    unsigned SubSize = TRI->getRegSizeInBits(SubReg, MRI) / 32;
    if (SubSize == 1) { // 32-bit sub-register
      SubRegs.push_back(SubReg);
    }
  }

  if (SubRegs.empty()) {
    // If no 32-bit sub-registers found, DstReg itself might be 32-bit
    if (DstSize == 1)
      SubRegs.push_back(BaseReg);
    else
      return false;
  }

  // Track which sub-registers have been overwritten
  SmallDenseSet<MCRegister, 8> OverwrittenRegs;

  unsigned InstrCount = 0;
  for (auto It = std::next(CopyMI->getIterator());
       It != MBB->end() && InstrCount < LookAhead; ++It, ++InstrCount) {
    MachineInstr &MI = *It;

    // First check if DstReg is used - if so, the copy is not dead
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;

      if (MO.isUse() && TRI->regsOverlap(MO.getReg(), DstReg)) {
        // The destination is used before being completely overwritten - not dead
        return false;
      }
    }

    // Then check if parts of DstReg are defined (overwritten)
    for (const MachineOperand &MO : MI.defs()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;

      Register DefReg = MO.getReg();

      // Check if this definition overlaps with our destination
      if (!TRI->regsOverlap(DefReg, DstReg))
        continue;

      // If it's the exact register, it's fully overwritten
      if (DefReg == DstReg) {
        LLVM_DEBUG(dbgs() << "  Copy destination " << printReg(DstReg, TRI)
                          << " is overwritten before use by: " << MI);
        return true;
      }

      // Check which sub-registers are being overwritten
      for (MCRegister SubReg : SubRegs) {
        if (TRI->regsOverlap(DefReg, SubReg)) {
          OverwrittenRegs.insert(SubReg);
        }
      }

      // Check if all sub-registers have been overwritten
      if (OverwrittenRegs.size() == SubRegs.size()) {
        LLVM_DEBUG(dbgs() << "  Copy destination " << printReg(DstReg, TRI)
                          << " is fully overwritten before use\n");
        return true;
      }
    }
  }

  return false;
}

bool SILoopCarriedCopyOpt::eliminateRedundantSGPRCopies(MachineLoop *Loop) {
  if (!EnableSGPRCopyElim)
    return false;

  MachineBasicBlock *Latch = Loop->getLoopLatch();
  MachineBasicBlock *Header = Loop->getHeader();

  if (!Latch || !Header)
    return false;

  // For single-block loops, header == latch
  // We want to find SGPR-to-VGPR copies where:
  // 1. The same SGPR is copied to the same VGPR multiple times
  // 2. The SGPR source is loop-invariant (not modified in the loop)
  // 3. The VGPR destination is killed after each use

  // Collect all SGPR-to-VGPR copies in the loop
  SmallVector<MachineInstr *, 64> SGPRCopies;
  DenseSet<MCRegister> ModifiedSGPRs;

  for (MachineBasicBlock *BB : Loop->blocks()) {
    for (MachineInstr &MI : *BB) {
      // Track SGPR modifications
      for (const MachineOperand &MO : MI.defs()) {
        if (MO.isReg() && MO.getReg().isPhysical() &&
            TRI->isSGPRReg(*MRI, MO.getReg())) {
          ModifiedSGPRs.insert(MO.getReg());
        }
      }

      if (isSGPRToVGPRMove(MI))
        SGPRCopies.push_back(&MI);
    }
  }

  if (SGPRCopies.empty())
    return false;

  LLVM_DEBUG(dbgs() << "Found " << SGPRCopies.size()
                    << " SGPR-to-VGPR copies in loop\n");

  bool Changed = false;

  // First pass: eliminate dead copies where the destination is overwritten
  // before being used
  SmallVector<MachineInstr *, 16> DeadCopies;
  for (MachineInstr *MI : SGPRCopies) {
    if (isOverwrittenBeforeUse(MI, TRI)) {
      DeadCopies.push_back(MI);
    }
  }

  for (MachineInstr *MI : DeadCopies) {
    LLVM_DEBUG(dbgs() << "  Eliminating dead SGPR copy: " << *MI);
    MI->eraseFromParent();
    Changed = true;
  }

  // Remove dead copies from the SGPRCopies list
  SGPRCopies.erase(
      llvm::remove_if(SGPRCopies,
                      [&DeadCopies](MachineInstr *MI) {
                        return llvm::is_contained(DeadCopies, MI);
                      }),
      SGPRCopies.end());

  if (SGPRCopies.empty())
    return Changed;

  // Group copies by (SrcReg, DstReg) pair
  // Only consider loop-invariant SGPR sources
  DenseMap<SGPRCopyKey, SmallVector<MachineInstr *, 4>> CopyGroups;

  for (MachineInstr *MI : SGPRCopies) {
    unsigned SrcReg = MI->getOperand(1).getReg();
    unsigned DstReg = MI->getOperand(0).getReg();

    // Skip if SGPR source is modified in the loop
    bool SrcModified = false;
    for (MCRegister Modified : ModifiedSGPRs) {
      if (TRI->regsOverlap(SrcReg, Modified)) {
        SrcModified = true;
        break;
      }
    }
    if (SrcModified) {
      LLVM_DEBUG(dbgs() << "  Skipping copy with modified SGPR: " << *MI);
      continue;
    }

    SGPRCopyKey Key{SrcReg, DstReg};
    CopyGroups[Key].push_back(MI);
  }

  // For each group with multiple copies, try to eliminate redundant ones
  for (auto &KV : CopyGroups) {
    SmallVector<MachineInstr *, 4> &Copies = KV.second;

    if (Copies.size() < 2)
      continue;

    LLVM_DEBUG({
      dbgs() << "Found " << Copies.size()
             << " copies of same SGPR->VGPR pair:\n";
      for (MachineInstr *MI : Copies)
        dbgs() << "  " << *MI;
    });

    // Sort copies by their position in the block
    // (For simplicity, we only handle single-block loops here)
    if (Header != Latch)
      continue;

    // Find the first copy and check if we can keep the VGPR live through
    // subsequent uses
    MachineInstr *FirstCopy = Copies[0];
    unsigned DstReg = FirstCopy->getOperand(0).getReg();

    // Check if DstReg is defined between copies (other than by the copies)
    // If so, we can't eliminate redundant copies
    bool CanEliminate = true;
    for (size_t i = 1; i < Copies.size() && CanEliminate; ++i) {
      MachineInstr *PrevCopy = Copies[i - 1];
      MachineInstr *CurrCopy = Copies[i];

      // Check for definitions of DstReg between PrevCopy and CurrCopy
      for (auto It = std::next(PrevCopy->getIterator());
           &*It != CurrCopy && CanEliminate; ++It) {
        for (const MachineOperand &MO : It->defs()) {
          if (MO.isReg() && MO.getReg().isPhysical() &&
              TRI->regsOverlap(MO.getReg(), DstReg)) {
            // DstReg is redefined between copies - can't eliminate
            LLVM_DEBUG(dbgs() << "  DstReg redefined between copies by: "
                              << *It);
            CanEliminate = false;
            break;
          }
        }
      }
    }

    if (!CanEliminate)
      continue;

    // Eliminate all copies after the first one
    for (size_t i = 1; i < Copies.size(); ++i) {
      MachineInstr *RedundantCopy = Copies[i];
      LLVM_DEBUG(dbgs() << "  Eliminating redundant copy: " << *RedundantCopy);
      RedundantCopy->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

bool SILoopCarriedCopyOpt::processLoop(MachineLoop *Loop) {
  bool Changed = false;

  // Get the latch block (block with back-edge to header)
  MachineBasicBlock *Latch = Loop->getLoopLatch();
  if (!Latch) {
    LLVM_DEBUG(dbgs() << "No latch block for loop\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Processing loop with latch: " << Latch->getName() << "\n");

  // First try to eliminate redundant SGPR-to-VGPR copies
  Changed |= eliminateRedundantSGPRCopies(Loop);

  // Collect VGPR moves to process from the entire latch block
  // These are copies that pass values to the next loop iteration
  SmallVector<MachineInstr *, 16> Moves;

  for (MachineInstr &MI : *Latch) {
    if (isVGPRMove(MI)) {
      // Only interested in VGPR-to-VGPR moves
      const MachineOperand &DstOp = MI.getOperand(0);
      const MachineOperand &SrcOp = MI.getOperand(1);
      if (DstOp.isReg() && SrcOp.isReg() &&
          DstOp.getReg().isPhysical() && SrcOp.getReg().isPhysical() &&
          TRI->isVGPR(*MRI, DstOp.getReg()) &&
          TRI->isVGPR(*MRI, SrcOp.getReg())) {
        LLVM_DEBUG(dbgs() << "  Found VGPR move: " << MI);
        Moves.push_back(&MI);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  Found " << Moves.size() << " VGPR moves\n");

  for (MachineInstr *Move : Moves) {
    LLVM_DEBUG(dbgs() << "Processing move: " << *Move);
    if (tryEliminateCopy(*Move, Loop))
      Changed = true;
  }

  // Process nested loops
  for (MachineLoop *SubLoop : *Loop)
    Changed |= processLoop(SubLoop);

  return Changed;
}

bool SILoopCarriedCopyOpt::run(MachineFunction &MF_) {
  if (!EnableLoopCarriedCopyOpt)
    return false;

  MF = &MF_;
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MRI = &MF->getRegInfo();

  bool Changed = false;

  for (MachineLoop *L : *MLI)
    Changed |= processLoop(L);

  return Changed;
}
