//===- HexagonInstPrinter.cpp - Convert Hexagon MCInst to assembly syntax -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints an Hexagon MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define GET_INSTRUCTION_NAME
#include "HexagonGenAsmWriter.inc"

void HexagonInstPrinter::printRegName(raw_ostream &O, MCRegister Reg) {
  O << getRegisterName(Reg);
}

void HexagonInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                   StringRef Annot, const MCSubtargetInfo &STI,
                                   raw_ostream &OS) {
  if (HexagonMCInstrInfo::isDuplex(MII, *MI)) {
    printInstruction(MI->getOperand(1).getInst(), Address, OS);
    OS << '\v';
    HasExtender = false;
    printInstruction(MI->getOperand(0).getInst(), Address, OS);
  } else {
    printInstruction(MI, Address, OS);
  }
  HasExtender = HexagonMCInstrInfo::isImmext(*MI);
  if ((MI->getOpcode() & HexagonII::INST_PARSE_MASK) ==
      HexagonII::INST_PARSE_PACKET_END)
    HasExtender = false;
}

void HexagonInstPrinter::printOperand(MCInst const *MI, unsigned OpNo,
                                      raw_ostream &O) const {
  if (HexagonMCInstrInfo::getExtendableOp(MII, *MI) == OpNo &&
      (HasExtender || HexagonMCInstrInfo::isConstExtended(MII, *MI)))
    O << "#";
  MCOperand const &MO = MI->getOperand(OpNo);
  if (MO.isReg()) {
    O << getRegisterName(MO.getReg());
  } else if (MO.isExpr()) {
    int64_t Value;
    if (MO.getExpr()->evaluateAsAbsolute(Value))
      O << formatImm(Value);
    else
      MAI.printExpr(O, *MO.getExpr());
  } else {
    llvm_unreachable("Unknown operand");
  }
}

void HexagonInstPrinter::printBrtarget(MCInst const *MI, unsigned OpNo,
                                       raw_ostream &O) const {
  MCOperand const &MO = MI->getOperand(OpNo);
  assert (MO.isExpr());
  MCExpr const &Expr = *MO.getExpr();
  int64_t Value;
  if (Expr.evaluateAsAbsolute(Value))
    O << format("0x%" PRIx64, Value);
  else {
    if (HasExtender || HexagonMCInstrInfo::isConstExtended(MII, *MI))
      if (HexagonMCInstrInfo::getExtendableOp(MII, *MI) == OpNo)
        O << "##";
    MAI.printExpr(O, Expr);
  }
}
